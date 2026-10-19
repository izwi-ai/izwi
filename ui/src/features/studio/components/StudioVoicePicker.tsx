import { useId, useMemo, useState } from "react";
import { ChevronDown, Download, Loader2, Play, Square } from "lucide-react";

import { api, type ModelInfo, type SavedVoiceSummary } from "@/api";
import { useAudioPreview } from "@/components/audio/useAudioPreview";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useReadAloud } from "@/features/chat/hooks/useReadAloud";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";
import {
  isModelOnDevice,
  modelName,
  voiceForModel,
  voiceOptionKey,
  voiceOptions,
  type StudioVoice,
} from "@/features/studio/studio";
import { cn } from "@/lib/utils";

const VISIBLE_VOICES = 5;

export interface StudioModelDownloads {
  downloadProgress: ModelDownloadProgressMap;
  onDownload: (variant: string) => void;
}

/** Download the model a voice needs, with its progress. */
export function ModelDownloadNotice({
  model,
  downloads,
}: {
  model: ModelInfo;
  downloads: StudioModelDownloads;
}) {
  const progress = downloads.downloadProgress[model.variant];
  const downloading = model.status === "downloading" || Boolean(progress);
  return (
    <div className="flex items-center gap-3 rounded-[12px] border border-[var(--border-subtle)] bg-[var(--surface-panel)] px-3.5 py-2.5 text-[13px]">
      <span className="min-w-0 flex-1 text-[var(--text-secondary)]">
        {downloading
          ? `Downloading ${modelName(model.variant, [model])}…${progress ? ` ${Math.round(progress.percent)}%` : ""}`
          : `${modelName(model.variant, [model])} isn't downloaded yet. Paragraphs render once it is.`}
      </span>
      {downloading ? (
        <Loader2 aria-hidden="true" className="h-4 w-4 shrink-0 animate-spin text-[var(--text-muted)]" />
      ) : (
        <button
          type="button"
          onClick={() => downloads.onDownload(model.variant)}
          className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-full bg-[var(--accent-solid)] px-3.5 text-[13px] font-medium text-[var(--text-on-accent)] transition-opacity hover:opacity-90"
        >
          <Download className="h-3.5 w-3.5" />
          Download
        </button>
      )}
    </div>
  );
}

interface StudioVoicePickerProps {
  models: ModelInfo[];
  savedVoices: SavedVoiceSummary[];
  value: StudioVoice | null;
  onChange: (voice: StudioVoice) => void;
  onError: (message: string) => void;
  downloads: StudioModelDownloads;
  disabled?: boolean;
}

/** A model and one of its voices, with previews. */
export function StudioVoicePicker({
  models,
  savedVoices,
  value,
  onChange,
  onError,
  downloads,
  disabled = false,
}: StudioVoicePickerProps) {
  const groupId = useId();
  const [showAll, setShowAll] = useState(false);
  const readAloud = useReadAloud(onError);
  const savedPreview = useAudioPreview(() => onError("That voice's sample couldn't play."));
  const model = models.find((candidate) => candidate.variant === value?.modelId);
  const onDevice = isModelOnDevice(model);
  const options = useMemo(() => voiceOptions(model, savedVoices), [model, savedVoices]);
  const selectedKey = value ? voiceOptionKey(value) : null;

  const visible = useMemo(() => {
    if (showAll || options.length <= VISIBLE_VOICES) return options;
    const first = options.slice(0, VISIBLE_VOICES);
    const selected = options.find((option) => option.key === selectedKey);
    return selected && !first.includes(selected)
      ? [selected, ...first.slice(0, VISIBLE_VOICES - 1)]
      : first;
  }, [options, selectedKey, showAll]);

  const stopPreviews = () => {
    readAloud.stop();
    savedPreview.stop();
  };

  return (
    <div className="flex flex-col gap-3">
      <Select
        value={value?.modelId ?? ""}
        disabled={disabled}
        onValueChange={(variant) => {
          stopPreviews();
          const next = voiceForModel(
            value,
            models.find((candidate) => candidate.variant === variant),
            savedVoices,
          );
          if (next) onChange(next);
        }}
      >
        <SelectTrigger
          aria-label="Model"
          className="h-9 w-full rounded-[10px] border-[var(--border-default)] bg-[var(--surface-panel)] text-[13.5px]"
        >
          <SelectValue placeholder="Choose a model" />
        </SelectTrigger>
        <SelectContent>
          {models.map((candidate) => (
            <SelectItem key={candidate.variant} value={candidate.variant}>
              {modelName(candidate.variant, models)}
              {isModelOnDevice(candidate) ? "" : " · not downloaded"}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {model && !onDevice ? <ModelDownloadNotice model={model} downloads={downloads} /> : null}

      {options.length === 0 ? (
        <p className="text-[13px] text-[var(--text-muted)]">
          {model?.speech_capabilities?.supports_reference_voice
            ? "Save a designed or cloned voice in Voices to use this model."
            : "This model has no voices to choose from."}
        </p>
      ) : (
        <div
          role="radiogroup"
          aria-label="Voice"
          id={groupId}
          className="flex flex-col overflow-hidden rounded-[12px] border border-[var(--border-subtle)]"
        >
          {visible.map((option) => {
            const checked = option.key === selectedKey;
            const previewing =
              readAloud.state?.messageId === option.key ||
              savedPreview.playingId === option.key;
            const loadingPreview =
              readAloud.state?.messageId === option.key && readAloud.state.status === "loading";
            return (
              <div
                key={option.key}
                className={cn(
                  "flex items-center gap-2 border-b border-[var(--border-subtle)] pl-3 pr-2 last:border-b-0",
                  checked && "bg-[var(--accent-soft)]",
                )}
              >
                <button
                  type="button"
                  role="radio"
                  aria-checked={checked}
                  aria-label={`${option.name}, ${option.detail}`}
                  disabled={disabled}
                  onClick={() =>
                    model &&
                    onChange({
                      modelId: model.variant,
                      voiceMode: option.voiceMode,
                      speaker: option.speaker,
                      savedVoiceId: option.savedVoiceId,
                    })
                  }
                  className="flex h-12 min-w-0 flex-1 items-center gap-3 text-left disabled:opacity-60"
                >
                  <span
                    aria-hidden="true"
                    className={cn(
                      "h-4 w-4 shrink-0 rounded-full",
                      checked
                        ? "border-[5px] border-[var(--accent-solid)]"
                        : "border-[1.5px] border-[var(--border-strong)]",
                    )}
                  />
                  <span className="flex min-w-0 flex-col">
                    <span className="truncate text-[13.5px] font-medium text-[var(--text-primary)]">
                      {option.name}
                    </span>
                    <span className="truncate text-xs text-[var(--text-muted)]">
                      {option.detail}
                    </span>
                  </span>
                </button>
                <button
                  type="button"
                  aria-label={previewing ? `Stop ${option.name}` : `Preview ${option.name}`}
                  disabled={!onDevice}
                  title={onDevice ? undefined : "Download the model to preview its voices"}
                  onClick={() => {
                    if (previewing) {
                      stopPreviews();
                      return;
                    }
                    stopPreviews();
                    if (option.savedVoiceId) {
                      savedPreview.toggle(option.key, api.savedVoiceAudioUrl(option.savedVoiceId));
                    } else if (model && option.speaker) {
                      readAloud.start(option.key, `Hi, I'm ${option.name}. This is how I sound.`, {
                        modelId: model.variant,
                        speaker: option.speaker,
                      });
                    }
                  }}
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-[var(--border-default)] text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-surface-2)] disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {loadingPreview ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : previewing ? (
                    <Square className="h-3 w-3 fill-current" />
                  ) : (
                    <Play className="h-3 w-3 fill-current" />
                  )}
                </button>
              </div>
            );
          })}
        </div>
      )}
      {options.length > VISIBLE_VOICES ? (
        <button
          type="button"
          onClick={() => setShowAll((current) => !current)}
          aria-expanded={showAll}
          aria-controls={groupId}
          className="flex h-8 items-center gap-1.5 self-start text-[13px] text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)]"
        >
          {showAll ? "Show fewer voices" : `Show all ${options.length} voices`}
          <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", showAll && "rotate-180")} />
        </button>
      ) : null}
    </div>
  );
}
