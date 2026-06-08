import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Check, Loader2, Sparkles } from "lucide-react";

import {
  api,
  type ModelInfo,
  type SavedVoiceSummary,
  type SpeechHistoryRecord,
  type SpeechHistoryRecordCreateRequest,
} from "@/api";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { StatusBadge } from "@/components/ui/status-badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getSpeakerProfilesForVariant } from "@/types";

interface NewTextToSpeechModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedModel: string | null;
  selectedModelInfo: ModelInfo | null;
  selectedModelReady: boolean;
  initialSavedVoiceId?: string | null;
  initialSpeaker?: string | null;
  onLoadSelectedModel?: (variant: string) => Promise<void> | void;
  onUnloadSelectedModel?: (variant: string) => Promise<void> | void;
  onOpenModelManager: () => void;
  onModelRequired: () => void;
  onCreated: (record: SpeechHistoryRecord) => Promise<void> | void;
  onStreamingStart?: () => void;
  onStreamingFinal?: (record: SpeechHistoryRecord) => void;
  onStreamingError?: (message: string) => void;
  onStreamingDone?: () => void;
}

type EffectiveVoiceWorkflow =
  | "saved_voice"
  | "built_in_voice"
  | "voice_description"
  | "unsupported";

function savedVoiceLabel(voice: SavedVoiceSummary): string {
  const source =
    voice.source_route_kind === "voice_cloning"
      ? "Clone"
      : voice.source_route_kind === "voice_design"
        ? "Design"
        : "Saved";
  return `${voice.name} (${source})`;
}

export function NewTextToSpeechModal({
  isOpen,
  onClose,
  selectedModel,
  selectedModelInfo,
  selectedModelReady,
  initialSavedVoiceId = null,
  initialSpeaker = null,
  onLoadSelectedModel,
  onUnloadSelectedModel,
  onOpenModelManager,
  onModelRequired,
  onCreated,
  onStreamingStart,
  onStreamingFinal,
  onStreamingError,
  onStreamingDone,
}: NewTextToSpeechModalProps) {
  const [text, setText] = useState("");
  const [speaker, setSpeaker] = useState(initialSpeaker || "Vivian");
  const [savedVoiceId, setSavedVoiceId] = useState(initialSavedVoiceId || "");
  const [voiceDescription, setVoiceDescription] = useState("");
  const [streamingEnabled, setStreamingEnabled] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [modelActionPending, setModelActionPending] = useState<
    "load" | "unload" | null
  >(null);
  const [error, setError] = useState<string | null>(null);
  const [savedVoices, setSavedVoices] = useState<SavedVoiceSummary[]>([]);
  const [savedVoicesLoading, setSavedVoicesLoading] = useState(false);
  const [savedVoicesError, setSavedVoicesError] = useState<string | null>(null);
  const wasOpenRef = useRef(false);

  const capabilities = selectedModelInfo?.speech_capabilities ?? null;
  const supportsBuiltInVoices = capabilities?.supports_builtin_voices ?? false;
  const supportsReferenceVoices =
    capabilities?.supports_reference_voice ?? false;
  const supportsVoiceDescription =
    capabilities?.supports_voice_description ?? false;
  const supportsStreaming = capabilities?.supports_streaming ?? false;
  const effectiveVoiceWorkflow: EffectiveVoiceWorkflow = supportsReferenceVoices
    ? "saved_voice"
    : supportsBuiltInVoices
      ? "built_in_voice"
      : supportsVoiceDescription
        ? "voice_description"
        : "unsupported";
  const usesSavedVoiceSelection = effectiveVoiceWorkflow === "saved_voice";
  const usesBuiltInVoiceSelection =
    effectiveVoiceWorkflow === "built_in_voice";
  const usesVoiceDescription = effectiveVoiceWorkflow === "voice_description";
  const streamAvailable = supportsStreaming;

  const speakerOptions = useMemo(
    () =>
      usesBuiltInVoiceSelection
        ? getSpeakerProfilesForVariant(selectedModel).map((profile) => ({
            id: profile.id,
            label: profile.name,
          }))
        : [],
    [selectedModel, usesBuiltInVoiceSelection],
  );
  const resolvedBuiltInSpeaker = useMemo(() => {
    if (!usesBuiltInVoiceSelection) {
      return "";
    }
    if (speakerOptions.some((option) => option.id === speaker)) {
      return speaker;
    }
    return speakerOptions[0]?.id ?? "";
  }, [speaker, speakerOptions, usesBuiltInVoiceSelection]);

  const resetModalState = useCallback(() => {
    setText("");
    setSpeaker(initialSpeaker || "Vivian");
    setSavedVoiceId(initialSavedVoiceId || "");
    setVoiceDescription("");
    setStreamingEnabled(true);
    setIsSubmitting(false);
    setModelActionPending(null);
    setError(null);
    setSavedVoices([]);
    setSavedVoicesLoading(false);
    setSavedVoicesError(null);
  }, [initialSavedVoiceId, initialSpeaker]);

  const handleModelAction = useCallback(async () => {
    if (!selectedModel || modelActionPending) {
      if (!selectedModel) {
        onModelRequired();
      }
      return;
    }

    setError(null);

    if (selectedModelReady) {
      if (!onUnloadSelectedModel) {
        return;
      }
      try {
        const result = onUnloadSelectedModel(selectedModel);
        if (result && typeof (result as Promise<void>).then === "function") {
          setModelActionPending("unload");
          await result;
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to unload model.");
      } finally {
        setModelActionPending(null);
      }
      return;
    }

    if (!onLoadSelectedModel) {
      onModelRequired();
      return;
    }

    try {
      const result = onLoadSelectedModel(selectedModel);
      if (result && typeof (result as Promise<void>).then === "function") {
        setModelActionPending("load");
        await result;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load model.");
    } finally {
      setModelActionPending(null);
    }
  }, [
    modelActionPending,
    onLoadSelectedModel,
    onModelRequired,
    onUnloadSelectedModel,
    selectedModel,
    selectedModelReady,
  ]);

  useEffect(() => {
    const wasOpen = wasOpenRef.current;
    if ((isOpen && !wasOpen) || (!isOpen && wasOpen)) {
      resetModalState();
    }
    wasOpenRef.current = isOpen;
  }, [isOpen, resetModalState]);

  useEffect(() => {
    if (!usesBuiltInVoiceSelection) {
      return;
    }
    if (!speakerOptions.length) {
      return;
    }
    if (speakerOptions.some((option) => option.id === speaker)) {
      return;
    }
    setSpeaker(speakerOptions[0].id);
  }, [speaker, speakerOptions, usesBuiltInVoiceSelection]);

  useEffect(() => {
    if (!isOpen || !usesSavedVoiceSelection) {
      return;
    }

    let cancelled = false;
    setSavedVoicesLoading(true);
    setSavedVoicesError(null);

    api
      .listSavedVoices()
      .then((voices) => {
        if (cancelled) {
          return;
        }
        setSavedVoices(voices);
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setSavedVoicesError(
          err instanceof Error ? err.message : "Failed to load saved voices.",
        );
      })
      .finally(() => {
        if (!cancelled) {
          setSavedVoicesLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [isOpen, usesSavedVoiceSelection]);

  const submitGeneration = useCallback(async () => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      setError("Select and load a compatible model before starting generation.");
      return;
    }

    const trimmedText = text.trim();
    if (!trimmedText) {
      setError("Enter text to generate speech.");
      return;
    }

    if (effectiveVoiceWorkflow === "unsupported") {
      setError(
        "The selected model does not expose built-in voices, saved voices, or voice direction prompts.",
      );
      return;
    }

    const trimmedVoiceDescription = voiceDescription.trim();

    if (usesVoiceDescription && !trimmedVoiceDescription) {
      setError("Add voice direction before creating this generation.");
      return;
    }

    if (usesSavedVoiceSelection && !savedVoiceId) {
      setError("Select a saved voice before starting generation.");
      return;
    }

    if (usesBuiltInVoiceSelection && !resolvedBuiltInSpeaker) {
      setError("Select a built-in voice before starting generation.");
      return;
    }

    const request: SpeechHistoryRecordCreateRequest = {
      model_id: selectedModel,
      text: trimmedText,
      speaker: usesBuiltInVoiceSelection ? resolvedBuiltInSpeaker : undefined,
      saved_voice_id: usesSavedVoiceSelection ? savedVoiceId : undefined,
      voice_description: usesVoiceDescription ? trimmedVoiceDescription : undefined,
    };

    setIsSubmitting(true);
    setError(null);

    try {
      const createdRecord =
        streamingEnabled && streamAvailable
          ? await new Promise<SpeechHistoryRecord>((resolve, reject) => {
              let settled = false;
              let streamRecord: SpeechHistoryRecord | null = null;

              api.createTextToSpeechRecordStream(request, {
                onCreated: (record) => {
                  streamRecord = record;
                  if (settled) {
                    return;
                  }
                  settled = true;
                  resolve(record);
                },
                onStart: () => {
                  onStreamingStart?.();
                },
                onFinal: ({ record }) => {
                  streamRecord = record;
                  onStreamingFinal?.(record);
                  if (settled) {
                    return;
                  }
                  settled = true;
                  resolve(record);
                },
                onError: (message) => {
                  onStreamingError?.(message);
                  if (settled) {
                    return;
                  }
                  settled = true;
                  reject(new Error(message));
                },
                onDone: () => {
                  onStreamingDone?.();
                  if (settled) {
                    return;
                  }
                  if (streamRecord) {
                    settled = true;
                    resolve(streamRecord);
                    return;
                  }
                  settled = true;
                  reject(
                    new Error(
                      "Generation started but no record was returned by the stream.",
                    ),
                  );
                },
              });
            })
          : await api.createTextToSpeechRecord(request);

      await onCreated(createdRecord);
      onClose();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to create generation.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }, [
    onClose,
    onCreated,
    onModelRequired,
    onStreamingDone,
    onStreamingError,
    onStreamingFinal,
    onStreamingStart,
    effectiveVoiceWorkflow,
    resolvedBuiltInSpeaker,
    savedVoiceId,
    selectedModel,
    selectedModelReady,
    streamAvailable,
    streamingEnabled,
    text,
    voiceDescription,
    usesBuiltInVoiceSelection,
    usesSavedVoiceSelection,
    usesVoiceDescription,
  ]);

  const modelStatus = selectedModelReady ? "ready" : "not_ready";
  const modelStatusCode = selectedModelInfo?.status ?? null;
  const modelStatusBusy =
    modelStatusCode === "loading" || modelStatusCode === "downloading";
  const modelActionBusy = modelStatusBusy || modelActionPending !== null;
  const isUnloadAction = selectedModelReady || modelActionPending === "unload";
  const modelActionLabel = modelStatusCode === "downloading"
    ? "Downloading model..."
    : modelStatusCode === "loading"
      ? "Loading model..."
      : modelActionPending === "load"
        ? "Loading model..."
        : modelActionPending === "unload"
          ? "Unloading model..."
          : isUnloadAction
            ? "Unload model"
            : "Load model";
  const modelActionButtonClass = isUnloadAction
    ? "mt-3 h-9 w-full gap-2 border-[var(--danger-border)] bg-[var(--danger-bg)] text-[var(--danger-text)] hover:bg-[var(--danger-bg-hover)] hover:text-[var(--danger-text)]"
    : "mt-3 h-9 w-full gap-2";

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-[52rem] overflow-hidden border-[var(--border-strong)] bg-[var(--bg-surface-0)] p-0">
        <div className="border-b border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-5 py-5 sm:px-6">
          <DialogTitle className="text-xl font-semibold tracking-tight text-[var(--text-primary)]">
            New text-to-speech job
          </DialogTitle>
          <DialogDescription className="mt-1 max-w-3xl text-[13px] leading-5 text-[var(--text-muted)]">
            Create a generation job and open it on its own record page
            immediately.
          </DialogDescription>
        </div>

        <div className="grid gap-0 lg:grid-cols-[minmax(0,1.15fr),minmax(18rem,0.85fr)]">
          <div className="border-b border-[var(--border-muted)] px-5 py-5 sm:px-6 lg:border-b-0 lg:border-r">
            <div className="mb-3 flex items-start justify-between gap-3">
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
                  Script
                </div>
                <h3 className="mt-1.5 text-xl font-semibold tracking-tight text-[var(--text-primary)]">
                  Enter text for generation
                </h3>
              </div>
              {isSubmitting ? (
                <div className="inline-flex items-center gap-2 rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-1.5 text-xs font-medium text-[var(--text-muted)]">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Creating job
                </div>
              ) : null}
            </div>

            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              disabled={isSubmitting}
              rows={10}
              placeholder="Write the text to speak..."
              className="min-h-[15rem] w-full resize-y rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-3 text-sm leading-6 text-[var(--text-primary)] outline-none transition-colors placeholder:text-[var(--text-muted)] focus:border-[var(--status-info-text)]"
            />

            <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
              <div className="text-xs text-[var(--text-muted)]">
                {Array.from(text).length} characters
              </div>
              <Button
                type="button"
                size="sm"
                className="h-9 gap-2"
                onClick={() => void submitGeneration()}
                disabled={isSubmitting}
              >
                {isSubmitting ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Sparkles className="h-4 w-4" />
                )}
                Create generation
              </Button>
            </div>
          </div>

          <div className="px-5 py-5 sm:px-6">
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
              Configuration
            </div>
            <h3 className="mt-1.5 text-xl font-semibold tracking-tight text-[var(--text-primary)]">
              Review settings
            </h3>

            <div className="mt-4 space-y-2.5">
              <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3.5">
                <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                  Model readiness
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <StatusBadge
                    tone={modelStatus === "ready" ? "success" : "warning"}
                  >
                    {modelStatus === "ready" ? "Ready" : "Not ready"}
                  </StatusBadge>
                  <span className="text-xs text-[var(--text-muted)]">
                    {selectedModel || "No model selected"}
                  </span>
                </div>
                <Button
                  type="button"
                  variant={isUnloadAction ? "outline" : "default"}
                  size="sm"
                  className={modelActionButtonClass}
                  onClick={() => void handleModelAction()}
                  disabled={isSubmitting || !selectedModel || modelActionBusy}
                >
                  {modelActionBusy ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : null}
                  {modelActionLabel}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="mt-2 h-9 w-full gap-2"
                  onClick={onOpenModelManager}
                  disabled={isSubmitting}
                >
                  Open models
                </Button>
              </div>

              {usesBuiltInVoiceSelection ? (
                <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3.5">
                  <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                    Built-in voice
                  </div>
                  <Select
                    value={speaker}
                    onValueChange={setSpeaker}
                    disabled={isSubmitting}
                  >
                    <SelectTrigger className="h-10 w-full rounded-2xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-sm">
                      <SelectValue placeholder="Select built-in voice" />
                    </SelectTrigger>
                    <SelectContent>
                      {speakerOptions.map((option) => (
                        <SelectItem key={option.id} value={option.id}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              ) : null}

              {usesSavedVoiceSelection ? (
                <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3.5">
                  <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                    Saved voice
                  </div>
                  <div className="space-y-2">
                    <Select
                      value={savedVoiceId}
                      onValueChange={setSavedVoiceId}
                      disabled={isSubmitting || savedVoicesLoading}
                    >
                      <SelectTrigger className="h-10 w-full rounded-2xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-sm">
                        <SelectValue placeholder="Select saved voice" />
                      </SelectTrigger>
                      <SelectContent>
                        {savedVoices.map((voice) => (
                          <SelectItem key={voice.id} value={voice.id}>
                            {savedVoiceLabel(voice)}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    {savedVoicesLoading ? (
                      <div className="text-xs text-[var(--text-muted)]">
                        Loading saved voices...
                      </div>
                    ) : null}
                    {savedVoicesError ? (
                      <div className="text-xs text-[var(--danger-text)]">
                        {savedVoicesError}
                      </div>
                    ) : null}
                  </div>
                </div>
              ) : null}

              {usesVoiceDescription ? (
                <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3.5">
                  <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                    Voice direction
                  </div>
                  <Input
                    value={voiceDescription}
                    onChange={(event) => setVoiceDescription(event.target.value)}
                    disabled={isSubmitting}
                    placeholder="Optional style guidance"
                  />
                </div>
              ) : null}

              <label className="flex items-start justify-between gap-3 rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3.5">
                <div className="min-w-0">
                  <div className="text-sm font-semibold text-[var(--text-primary)]">
                    Stream results
                  </div>
                  <div className="mt-0.5 text-[13px] leading-5 text-[var(--text-muted)]">
                    Send live chunks when model streaming is available.
                  </div>
                </div>
                <div className="relative mt-0.5 shrink-0">
                  <input
                    type="checkbox"
                    checked={streamingEnabled}
                    onChange={(event) => setStreamingEnabled(event.target.checked)}
                    className="peer sr-only"
                    disabled={isSubmitting || !streamAvailable}
                  />
                  <span className="flex h-5 w-5 items-center justify-center rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface-0)] text-white transition peer-checked:border-[var(--status-info-text)] peer-checked:bg-[var(--status-info-text)] peer-focus-visible:outline-none peer-focus-visible:ring-2 peer-focus-visible:ring-ring/45 peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-background peer-disabled:opacity-50">
                    <Check className="h-3.5 w-3.5 opacity-0 transition peer-checked:opacity-100" />
                  </span>
                </div>
              </label>
            </div>

            {error ? (
              <div className="mt-4 rounded-xl border border-[var(--danger-border)] bg-[var(--danger-bg)] p-3 text-sm text-[var(--danger-text)]">
                <div className="flex items-start gap-2">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>{error}</span>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
