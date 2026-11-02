import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Dices, Loader2, Sparkles } from "lucide-react";

import {
  api,
  type ModelInfo,
  type SavedVoiceSummary,
  type SpeechControlId,
  type SpeechControlRange,
  type SpeechHistoryRecord,
  type SpeechHistoryRecordCreateRequest,
  type TTSRequest,
} from "@/api";
import { useAudioPreview } from "@/components/audio/useAudioPreview";
import { CreationDialog } from "@/components/CreationDialog";
import {
  CreationBody,
  CreationField,
  CreationOption,
  ModelRow,
  ModelRowAction,
} from "@/components/creation/CreationParts";
import {
  SegmentedFilter,
  type SegmentOption,
} from "@/components/history/HistoryToolbar";
import { getModelDisplayName } from "@/features/models/catalog/modelMetadata";
import { DialogueEditor } from "@/features/text-to-speech/components/DialogueEditor";
import {
  FISH_DRAMA3_SCOPED_TAGS,
  dialogueLimits,
  dialogueProblem,
  dialogueReferences,
  dialogueScript,
  insertTagAt,
  newDialogueScene,
  type DialogueLine,
  type DialogueScene,
} from "@/features/text-to-speech/dialogue";
import { savedVoiceLabel } from "@/features/text-to-speech/support";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  getSpeakerProfilesForVariant,
  isFishDrama3Variant,
  isMaya1Variant,
} from "@/types";

interface NewTextToSpeechModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedModel: string | null;
  selectedModelInfo: ModelInfo | null;
  selectedModelReady: boolean;
  isModelManagerOpen?: boolean;
  initialSavedVoiceId?: string | null;
  initialSpeaker?: string | null;
  /** Opens the dialog on the dialogue editor with these rows (a take's,
   * parsed back). */
  initialDialogue?: DialogueScene | null;
  onLoadSelectedModel?: (variant: string) => Promise<void> | void;
  onUnloadSelectedModel?: (variant: string) => Promise<void> | void;
  onOpenModelManager: () => void;
  onModelRequired: () => void;
  onCreated: (record: SpeechHistoryRecord) => Promise<void> | void;
  onCreateStream: (request: SpeechHistoryRecordCreateRequest) => Promise<SpeechHistoryRecord>;
}

type EffectiveVoiceWorkflow =
  | "saved_voice"
  | "built_in_voice"
  | "voice_description"
  | "unsupported";

const SPEECH_CONTROL_LABELS: Record<SpeechControlId, string> = {
  exaggeration: "Exaggeration",
  cfg_weight: "Guidance",
  min_p: "Min-p",
};

/** The controls a request sets: only values moved off the model's default. */
function changedSpeechControls(
  controls: SpeechControlRange[],
  values: Partial<Record<SpeechControlId, number>>,
): Partial<Record<SpeechControlId, number>> {
  const changed: Partial<Record<SpeechControlId, number>> = {};
  for (const range of controls) {
    const value = values[range.control];
    if (value !== undefined && Math.abs(value - range.default) > 0.001) {
      changed[range.control] = value;
    }
  }
  return changed;
}

/** The language a multilingual model is asked for until one is chosen. */
const DEFAULT_SPEECH_LANGUAGE = "en";

/** The picker's choice for a model that reads the language from the text's
 * script (Maya 2 Native): the request names none. */
const AUTO_DETECT_LANGUAGE = "auto";

/** The saved-voice picker's "no saved voice" choice, for a model that also
 * has built-in voices or a seeded voice (Radix Select takes no empty value). */
const BUILT_IN_VOICE = "__built_in__";

type SpeechMode = "single" | "dialogue";

const NO_TAGS: readonly string[] = [];

const MODE_OPTIONS: SegmentOption<SpeechMode>[] = [
  { id: "single", label: "Single voice" },
  { id: "dialogue", label: "Dialogue" },
];

/** A seed to start from, so a line preview and the scene share voices. */
function randomSeed(): number {
  return Math.floor(Math.random() * 1_000_000);
}

/** The seed field as a request sends it: none when empty, an error when
 * it is not a whole number. */
function parseSeed(value: string): { seed?: number; error?: string } {
  const trimmed = value.trim();
  if (!trimmed) {
    return {};
  }
  const seed = Number(trimmed);
  if (!/^\d+$/.test(trimmed) || !Number.isSafeInteger(seed)) {
    return { error: "The seed must be a whole number, 0 or more." };
  }
  return { seed };
}

export function NewTextToSpeechModal({
  isOpen,
  onClose,
  selectedModel,
  selectedModelInfo,
  selectedModelReady,
  isModelManagerOpen = false,
  initialSavedVoiceId = null,
  initialSpeaker = null,
  initialDialogue = null,
  onLoadSelectedModel,
  onUnloadSelectedModel,
  onOpenModelManager,
  onModelRequired,
  onCreated,
  onCreateStream,
}: NewTextToSpeechModalProps) {
  const [text, setText] = useState("");
  const [speaker, setSpeaker] = useState(initialSpeaker || "Vivian");
  const [savedVoiceId, setSavedVoiceId] = useState(initialSavedVoiceId || "");
  const [voiceDescription, setVoiceDescription] = useState("");
  const [speed, setSpeed] = useState(1);
  const [language, setLanguage] = useState<string | null>(null);
  const [controlValues, setControlValues] = useState<
    Partial<Record<SpeechControlId, number>>
  >({});
  const [streamingEnabled, setStreamingEnabled] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [modelActionPending, setModelActionPending] = useState<
    "load" | "unload" | null
  >(null);
  const [error, setError] = useState<string | null>(null);
  const [savedVoices, setSavedVoices] = useState<SavedVoiceSummary[]>([]);
  const [savedVoicesLoading, setSavedVoicesLoading] = useState(false);
  const [savedVoicesError, setSavedVoicesError] = useState<string | null>(null);
  const [mode, setMode] = useState<SpeechMode>(
    initialDialogue ? "dialogue" : "single",
  );
  const [scene, setScene] = useState<DialogueScene>(
    () => initialDialogue ?? newDialogueScene(null),
  );
  const [seed, setSeed] = useState("");
  // The latest line preview, replayed while its request is unchanged.
  const [linePreview, setLinePreview] = useState<{
    lineId: string;
    key: string;
    url: string;
  } | null>(null);
  const [previewingLineId, setPreviewingLineId] = useState<string | null>(
    null,
  );
  const previewRunRef = useRef(0);
  const linePreviewAudio = useAudioPreview(() =>
    setError("The preview could not play in this browser."),
  );
  const stopLinePreview = linePreviewAudio.stop;
  const wasOpenRef = useRef(false);

  const capabilities = selectedModelInfo?.speech_capabilities ?? null;
  // A model that speaks without voice input (Fish Audio Drama 3): the seed
  // sets the voice, so a saved voice is optional.
  const seededVoice = capabilities?.supports_seeded_voice ?? false;
  const limits = useMemo(() => dialogueLimits(capabilities), [capabilities]);
  const dialogueMode = mode === "dialogue" && limits !== null;
  const scopedTags =
    selectedModel && isFishDrama3Variant(selectedModel)
      ? FISH_DRAMA3_SCOPED_TAGS
      : NO_TAGS;
  const supportsSpeed = capabilities?.supports_speed_control ?? false;
  const supportsBuiltInVoices = capabilities?.supports_builtin_voices ?? false;
  const supportsReferenceVoices =
    capabilities?.supports_reference_voice ?? false;
  // Chatterbox clones from audio alone; every other reference model needs
  // the reference's words, so voices saved without them are left out.
  const transcriptRequired =
    capabilities?.reference_transcript_required ?? true;
  const supportsVoiceDescription =
    capabilities?.supports_voice_description ?? false;
  const supportsStreaming = capabilities?.supports_streaming ?? false;
  const speechControls = useMemo(
    () => capabilities?.controls ?? [],
    [capabilities],
  );
  // A model that speaks several named languages (Chatterbox Multilingual,
  // Maya 2 Native) is told which one, or left to read it from the text's
  // script where it can; one with a single language needs no picker.
  const languageAutoDetect = capabilities?.language_auto_detect ?? false;
  const speechLanguages = useMemo(() => {
    const named = [...(capabilities?.languages ?? [])].sort((a, b) =>
      a.name.localeCompare(b.name),
    );
    return capabilities?.language_auto_detect
      ? [{ code: AUTO_DETECT_LANGUAGE, name: "Auto-detect" }, ...named]
      : named;
  }, [capabilities]);
  const offersLanguage = speechLanguages.length > 1;
  const defaultLanguage =
    capabilities?.default_language ?? DEFAULT_SPEECH_LANGUAGE;
  const resolvedLanguage =
    language !== null &&
    speechLanguages.some((option) => option.code === language)
      ? language
      : !languageAutoDetect &&
          speechLanguages.some((option) => option.code === defaultLanguage)
        ? defaultLanguage
        : (speechLanguages[0]?.code ?? DEFAULT_SPEECH_LANGUAGE);
  const effectiveVoiceWorkflow: EffectiveVoiceWorkflow =
    supportsReferenceVoices && !supportsBuiltInVoices
      ? "saved_voice"
      : supportsBuiltInVoices
        ? "built_in_voice"
        : supportsVoiceDescription
          ? "voice_description"
          : "unsupported";
  const usesSavedVoiceSelection = effectiveVoiceWorkflow === "saved_voice";
  // A model with both (Chatterbox) speaks with its built-in voice unless a
  // saved voice is chosen.
  const offersSavedVoice = supportsReferenceVoices;
  const useSavedVoice = offersSavedVoice && Boolean(savedVoiceId);
  const usesBuiltInVoiceSelection =
    effectiveVoiceWorkflow === "built_in_voice";
  const usesVoiceDescription = effectiveVoiceWorkflow === "voice_description";
  // A built-in or seeded voice speaks when no saved voice is chosen.
  const savedVoiceOptional = usesBuiltInVoiceSelection || seededVoice;
  // A description is required only when it is the model's sole voice mode;
  // models with built-in voices (Qwen3 1.7B CustomVoice) accept it as
  // optional style direction alongside the speaker.
  const showsVoiceDescription =
    supportsVoiceDescription &&
    (usesVoiceDescription || usesBuiltInVoiceSelection);
  // Maya1's built-in voices are themselves descriptions: one of the user's
  // own replaces the voice rather than styling it.
  const descriptionReplacesVoice = selectedModel
    ? isMaya1Variant(selectedModel)
    : false;
  // Inline tags the script may carry (Maya1's <laugh>, Maya 2's [laughs], ...).
  const inlineTags = useMemo(
    () => capabilities?.inline_tags ?? [],
    [capabilities],
  );
  const scriptRef = useRef<HTMLTextAreaElement>(null);
  const insertInlineTag = useCallback(
    (tag: string) => {
      const field = scriptRef.current;
      const next = insertTagAt(
        text,
        field?.selectionStart ?? text.length,
        field?.selectionEnd ?? text.length,
        tag,
      );
      setText(next.text);
      requestAnimationFrame(() => {
        field?.focus();
        field?.setSelectionRange(next.caret, next.caret);
      });
    },
    [text],
  );
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
    setLanguage(null);
    setControlValues({});
    setStreamingEnabled(true);
    setIsSubmitting(false);
    setModelActionPending(null);
    setError(null);
    setSavedVoices([]);
    setSavedVoicesLoading(false);
    setSavedVoicesError(null);
    setMode(initialDialogue ? "dialogue" : "single");
    setScene(initialDialogue ?? newDialogueScene(limits));
    setSeed(String(randomSeed()));
    previewRunRef.current += 1;
    setPreviewingLineId(null);
    setLinePreview(null);
    stopLinePreview();
  }, [initialDialogue, initialSavedVoiceId, initialSpeaker, limits, stopLinePreview]);

  // A replaced or abandoned preview frees its audio.
  useEffect(() => {
    if (!linePreview) {
      return;
    }
    const { url } = linePreview;
    return () => URL.revokeObjectURL(url);
  }, [linePreview]);

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
    if (!isOpen || !offersSavedVoice) {
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
  }, [isOpen, offersSavedVoice]);

  const usableSavedVoices = useMemo(
    () =>
      transcriptRequired
        ? savedVoices.filter((voice) => voice.reference_text_chars > 0)
        : savedVoices,
    [savedVoices, transcriptRequired],
  );

  // Fields every request of the dialog carries besides its text and voices.
  const sharedRequestFields = useCallback(() => {
    const parsedSeed = seededVoice ? parseSeed(seed) : {};
    return {
      error: parsedSeed.error,
      fields: {
        language:
          offersLanguage && resolvedLanguage !== AUTO_DETECT_LANGUAGE
            ? resolvedLanguage
            : undefined,
        // Only a model with speed control gets one (Kokoro); 1× sends none.
        speed:
          supportsSpeed && Math.abs(speed - 1) > 0.001 ? speed : undefined,
        ...changedSpeechControls(speechControls, controlValues),
        seed: parsedSeed.seed,
      },
    };
  }, [
    controlValues,
    offersLanguage,
    resolvedLanguage,
    seed,
    seededVoice,
    speechControls,
    speed,
    supportsSpeed,
  ]);

  // The page's record path: streamed when the switch allows, else a job.
  const sendGeneration = useCallback(
    async (request: SpeechHistoryRecordCreateRequest) => {
      stopLinePreview();
      setIsSubmitting(true);
      setError(null);

      try {
        const createdRecord =
          streamingEnabled && streamAvailable
            ? await onCreateStream(request)
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
    },
    [
      onClose,
      onCreated,
      onCreateStream,
      stopLinePreview,
      streamAvailable,
      streamingEnabled,
    ],
  );

  const submitGeneration = useCallback(async () => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      setError("Select and load a compatible model before starting generation.");
      return;
    }

    // The whole scene: every line, and the saved voices of its speakers.
    if (dialogueMode && limits) {
      const problem = dialogueProblem(scene, scene.lines, limits);
      const shared = sharedRequestFields();
      if (problem || shared.error) {
        setError(problem ?? shared.error ?? null);
        return;
      }
      const references = dialogueReferences(scene, scene.lines);
      await sendGeneration({
        model_id: selectedModel,
        script: dialogueScript(scene.lines),
        references: references.length > 0 ? references : undefined,
        ...shared.fields,
      });
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

    // A seeded-voice model speaks without one: the seed sets the voice.
    if (usesSavedVoiceSelection && !savedVoiceId && !seededVoice) {
      setError("Select a saved voice before starting generation.");
      return;
    }

    if (usesBuiltInVoiceSelection && !useSavedVoice && !resolvedBuiltInSpeaker) {
      setError("Select a built-in voice before starting generation.");
      return;
    }

    const shared = sharedRequestFields();
    if (shared.error) {
      setError(shared.error);
      return;
    }

    await sendGeneration({
      model_id: selectedModel,
      text: trimmedText,
      speaker:
        usesBuiltInVoiceSelection && !useSavedVoice
          ? resolvedBuiltInSpeaker
          : undefined,
      saved_voice_id: useSavedVoice ? savedVoiceId : undefined,
      voice_description:
        showsVoiceDescription && trimmedVoiceDescription
          ? trimmedVoiceDescription
          : undefined,
      ...shared.fields,
    });
  }, [
    onModelRequired,
    dialogueMode,
    effectiveVoiceWorkflow,
    limits,
    resolvedBuiltInSpeaker,
    savedVoiceId,
    scene,
    seededVoice,
    selectedModel,
    selectedModelReady,
    sendGeneration,
    sharedRequestFields,
    text,
    voiceDescription,
    usesBuiltInVoiceSelection,
    usesSavedVoiceSelection,
    useSavedVoice,
    usesVoiceDescription,
    showsVoiceDescription,
  ]);

  // A line on its own, with its speaker's saved voice, through the speech
  // route: nothing is added to history.
  const previewLine = useCallback(
    async (line: DialogueLine) => {
      if (!selectedModel || !selectedModelReady) {
        onModelRequired();
        setError("Select and load a compatible model before previewing a line.");
        return;
      }
      if (!limits) {
        return;
      }
      const problem = dialogueProblem(scene, [line], limits);
      if (problem) {
        setError(problem);
        return;
      }
      const shared = sharedRequestFields();
      if (shared.error) {
        setError(shared.error);
        return;
      }
      const references = dialogueReferences(scene, [line]);
      const request: TTSRequest = {
        model_id: selectedModel,
        script: dialogueScript([line]),
        references: references.length > 0 ? references : undefined,
        language: shared.fields.language,
        speed: shared.fields.speed,
        seed: shared.fields.seed,
      };
      const key = JSON.stringify(request);
      if (linePreview?.lineId === line.id && linePreview.key === key) {
        linePreviewAudio.toggle(line.id, linePreview.url);
        return;
      }
      stopLinePreview();
      const run = ++previewRunRef.current;
      setPreviewingLineId(line.id);
      setError(null);
      try {
        const { audioBlob } = await api.generateTTSWithStats(request);
        if (run !== previewRunRef.current) {
          return;
        }
        const url = URL.createObjectURL(audioBlob);
        setLinePreview({ lineId: line.id, key, url });
        linePreviewAudio.toggle(line.id, url);
      } catch (err) {
        if (run === previewRunRef.current) {
          setError(
            err instanceof Error ? err.message : "Failed to preview the line.",
          );
        }
      } finally {
        if (run === previewRunRef.current) {
          setPreviewingLineId(null);
        }
      }
    },
    [
      limits,
      linePreview,
      linePreviewAudio,
      onModelRequired,
      scene,
      selectedModel,
      selectedModelReady,
      sharedRequestFields,
      stopLinePreview,
    ],
  );

  const sceneCharacters = scene.lines.reduce(
    (total, line) => total + Array.from(line.text).length,
    0,
  );
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

  return (
    <CreationDialog
      open={isOpen}
      onClose={onClose}
      modal={!isModelManagerOpen}
      blockDismiss={isModelManagerOpen}
      className="max-w-208"
      title="New text-to-speech job"
      description="Create a generation job and open it on its own record page immediately."
      header={
        limits ? (
          <div className="mt-3">
            <SegmentedFilter
              label="Speech mode"
              options={MODE_OPTIONS}
              value={dialogueMode ? "dialogue" : "single"}
              onChange={(next) => {
                stopLinePreview();
                setError(null);
                setMode(next);
              }}
              disabled={isSubmitting}
            />
          </div>
        ) : null
      }
      onSubmit={() => void submitGeneration()}
      submitLabel={dialogueMode ? "Generate scene" : "Create generation"}
      submitIcon={<Sparkles className="h-4 w-4" />}
      submitting={isSubmitting}
      footerStart={
        dialogueMode
          ? `${scene.lines.length} ${scene.lines.length === 1 ? "line" : "lines"} · ${sceneCharacters} characters`
          : `${Array.from(text).length} characters`
      }
    >
        <CreationBody
          main={
            dialogueMode && limits ? (
              <DialogueEditor
                scene={scene}
                onChange={setScene}
                limits={limits}
                savedVoices={savedVoices}
                savedVoicesLoading={savedVoicesLoading}
                savedVoicesError={savedVoicesError}
                pointTags={inlineTags}
                scopedTags={scopedTags}
                disabled={isSubmitting}
                previewingLineId={previewingLineId}
                playingLineId={linePreviewAudio.playingId}
                onPreviewLine={(line) => void previewLine(line)}
                restored={Boolean(initialDialogue)}
              />
            ) : (
              <>
                <textarea
                  ref={scriptRef}
                  aria-label="Script"
                  value={text}
                  onChange={(event) => setText(event.target.value)}
                  disabled={isSubmitting}
                  rows={12}
                  autoFocus
                  placeholder="Write or paste the text to speak…"
                  className="min-h-68 w-full flex-1 resize-y rounded-[12px] border border-(--border-default) bg-(--surface-panel) px-4 py-3 text-[14.5px] leading-6 text-(--text-primary) outline-hidden transition-colors placeholder:text-(--text-subtle) focus:border-(--border-strong)"
                />
                {inlineTags.length > 0 ? (
                  <Select
                    value=""
                    onValueChange={insertInlineTag}
                    disabled={isSubmitting}
                  >
                    <SelectTrigger
                      aria-label="Insert an expression tag"
                      className="h-8 w-fit gap-2 text-xs"
                    >
                      <SelectValue
                        placeholder={`Insert a tag: ${inlineTags.slice(0, 3).join(", ")}…`}
                      />
                    </SelectTrigger>
                    <SelectContent>
                      {inlineTags.map((tag) => (
                        <SelectItem key={tag} value={tag}>
                          {tag}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : null}
                {isSubmitting ? (
                  <div className="inline-flex items-center gap-2 text-xs text-(--text-muted)">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    Creating job
                  </div>
                ) : null}
              </>
            )
          }
          side={
            <>
              {offersLanguage ? (
                <CreationField label="Language">
                  <Select
                    value={resolvedLanguage}
                    onValueChange={setLanguage}
                    disabled={isSubmitting}
                  >
                    <SelectTrigger
                      aria-label="Language"
                      className="h-9 w-full rounded-[10px] border-(--border-default) bg-(--surface-panel) text-[13.5px]"
                    >
                      <SelectValue placeholder="Select language" />
                    </SelectTrigger>
                    <SelectContent>
                      {speechLanguages.map((option) => (
                        <SelectItem key={option.code} value={option.code}>
                          {option.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </CreationField>
              ) : null}

              {supportsSpeed ? (
                <CreationField
                  label="Speed"
                  hint={`${speed.toFixed(2)}×`}
                >
                  <Slider
                    aria-label="Speed"
                    min={0.5}
                    max={1.5}
                    step={0.05}
                    value={[speed]}
                    onValueChange={([next]) => setSpeed(next ?? 1)}
                    disabled={isSubmitting}
                  />
                </CreationField>
              ) : null}

              {speechControls.map((range) => {
                const value = controlValues[range.control] ?? range.default;
                const label = SPEECH_CONTROL_LABELS[range.control];
                return (
                  <CreationField
                    key={range.control}
                    label={label}
                    hint={value.toFixed(2)}
                  >
                    <Slider
                      aria-label={label}
                      min={range.min}
                      max={range.max}
                      step={0.05}
                      value={[value]}
                      onValueChange={([next]) =>
                        setControlValues((current) => ({
                          ...current,
                          [range.control]: next ?? range.default,
                        }))
                      }
                      disabled={isSubmitting}
                    />
                  </CreationField>
                );
              })}

              {usesBuiltInVoiceSelection && !dialogueMode ? (
                <CreationField label="Built-in voice">
                  <Select
                    value={speaker}
                    onValueChange={setSpeaker}
                    disabled={isSubmitting}
                  >
                    <SelectTrigger
                      aria-label="Built-in voice"
                      className="h-9 w-full rounded-[10px] border-(--border-default) bg-(--surface-panel) text-[13.5px]"
                    >
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
                </CreationField>
              ) : null}

              {offersSavedVoice && !dialogueMode ? (
                <CreationField
                  label={
                    savedVoiceOptional
                      ? "Saved voice (optional)"
                      : "Saved voice"
                  }
                  hint={
                    savedVoicesLoading ? (
                      "Loading saved voices…"
                    ) : savedVoicesError ? (
                      <span className="text-(--danger-text)">
                        {savedVoicesError}
                      </span>
                    ) : usesBuiltInVoiceSelection ? (
                      "A saved voice replaces the built-in one."
                    ) : seededVoice ? (
                      "Without one, the seed sets the voice. A saved voice needs its transcript."
                    ) : undefined
                  }
                >
                  <Select
                    value={
                      savedVoiceOptional
                        ? savedVoiceId || BUILT_IN_VOICE
                        : savedVoiceId
                    }
                    onValueChange={(value) =>
                      setSavedVoiceId(value === BUILT_IN_VOICE ? "" : value)
                    }
                    disabled={isSubmitting || savedVoicesLoading}
                  >
                    <SelectTrigger
                      aria-label="Saved voice"
                      className="h-9 w-full rounded-[10px] border-(--border-default) bg-(--surface-panel) text-[13.5px]"
                    >
                      <SelectValue placeholder="Select saved voice" />
                    </SelectTrigger>
                    <SelectContent>
                      {savedVoiceOptional ? (
                        <SelectItem value={BUILT_IN_VOICE}>
                          {usesBuiltInVoiceSelection
                            ? "None (built-in voice)"
                            : "None (seeded voice)"}
                        </SelectItem>
                      ) : null}
                      {usableSavedVoices.map((voice) => (
                        <SelectItem key={voice.id} value={voice.id}>
                          {savedVoiceLabel(voice)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </CreationField>
              ) : null}

              {showsVoiceDescription && !dialogueMode ? (
                <CreationField
                  label={
                    usesVoiceDescription
                      ? "Voice direction"
                      : descriptionReplacesVoice
                        ? "Your own voice (optional)"
                        : "Style direction (optional)"
                  }
                  htmlFor="tts-voice-description"
                >
                  {descriptionReplacesVoice ? (
                    <textarea
                      id="tts-voice-description"
                      value={voiceDescription}
                      onChange={(event) =>
                        setVoiceDescription(event.target.value)
                      }
                      disabled={isSubmitting}
                      rows={4}
                      placeholder="Describe a voice to use instead, e.g. Realistic female voice in the 40s age with a british accent. Low pitch, warm timbre, slow pacing."
                      className="w-full resize-y rounded-[10px] border border-(--border-default) bg-(--surface-panel) px-3 py-2 text-[13px] leading-5 text-(--text-primary) outline-hidden transition-colors placeholder:text-(--text-subtle) focus:border-(--border-strong)"
                    />
                  ) : (
                    <Input
                      id="tts-voice-description"
                      value={voiceDescription}
                      onChange={(event) =>
                        setVoiceDescription(event.target.value)
                      }
                      disabled={isSubmitting}
                      required={usesVoiceDescription}
                      placeholder={
                        usesVoiceDescription
                          ? "Describe the voice, e.g. a calm, warm narrator"
                          : "e.g. speak slowly and warmly"
                      }
                    />
                  )}
                </CreationField>
              ) : null}

              {seededVoice ? (
                <CreationField
                  label="Seed"
                  htmlFor="tts-seed"
                  hint={
                    dialogueMode
                      ? "Speakers without a saved voice speak in voices the seed sets."
                      : "The same seed gives the same voice. Leave it empty for a new voice each time."
                  }
                >
                  <div className="flex items-center gap-2">
                    <Input
                      id="tts-seed"
                      inputMode="numeric"
                      value={seed}
                      onChange={(event) => setSeed(event.target.value)}
                      disabled={isSubmitting}
                      placeholder="Random"
                      className="h-9"
                    />
                    <Button
                      type="button"
                      variant="outline"
                      size="icon"
                      className="h-9 w-9 shrink-0"
                      aria-label="Pick a new seed"
                      onClick={() => setSeed(String(randomSeed()))}
                      disabled={isSubmitting}
                    >
                      <Dices aria-hidden="true" />
                    </Button>
                  </div>
                </CreationField>
              ) : null}

              <CreationField label="Model">
                <ModelRow
                  name={
                    selectedModel
                      ? getModelDisplayName(
                          selectedModel,
                          selectedModelInfo?.display_name,
                        )
                      : "No model selected"
                  }
                  status={
                    !selectedModel
                      ? "missing"
                      : modelActionBusy
                        ? "busy"
                        : modelStatus === "ready"
                          ? "ready"
                          : "not_ready"
                  }
                  statusLabel={
                    modelActionBusy
                      ? modelActionLabel.replace(/\.\.\.$/, "…")
                      : modelStatus === "ready"
                        ? "Loaded"
                        : selectedModel
                          ? "Not loaded"
                          : "Choose one in Models"
                  }
                  actions={
                    <>
                      {selectedModel && !modelActionBusy ? (
                        <ModelRowAction
                          onClick={() => void handleModelAction()}
                          disabled={isSubmitting}
                        >
                          {isUnloadAction ? "Unload" : "Load"}
                        </ModelRowAction>
                      ) : null}
                      <ModelRowAction
                        onClick={onOpenModelManager}
                        disabled={isSubmitting}
                        aria-label="Change model"
                      >
                        Change
                      </ModelRowAction>
                    </>
                  }
                />
              </CreationField>

              <CreationOption
                title="Stream results"
                description="Start listening before the full take is ready."
                // A final-only model shows streaming off, not a disabled "on".
                checked={streamingEnabled && streamAvailable}
                onCheckedChange={setStreamingEnabled}
                disabled={isSubmitting || !streamAvailable}
              />

              {error ? (
                <div
                  role="alert"
                  className="flex items-start gap-2 rounded-[10px] border border-(--danger-border) bg-(--danger-bg) p-2.5 text-[13px] text-(--danger-text)"
                >
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>{error}</span>
                </div>
              ) : null}
            </>
          }
        />
    </CreationDialog>
  );
}
