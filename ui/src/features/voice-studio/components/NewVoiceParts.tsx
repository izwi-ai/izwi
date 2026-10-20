import { AlertTriangle, CheckCircle2, Pause, Play } from "lucide-react";

import type { ModelInfo } from "@/api";
import {
  CreationField,
  ModelRow,
  ModelRowAction,
} from "@/components/creation/CreationParts";
import { getModelDisplayName } from "@/features/models/catalog/modelMetadata";
import { cn } from "@/lib/utils";

export const FIELD_LABEL_CLASS =
  "text-[12.5px] font-medium text-[var(--text-muted)]";

/** Overrides for Input so a one-line field matches the dialog's controls. */
export const TEXT_INPUT_CLASS =
  "h-9 rounded-[10px] border-[var(--border-default)] bg-[var(--surface-panel)] px-3 text-[13.5px] shadow-none";

export const TEXTAREA_CLASS =
  "w-full resize-none rounded-[12px] border border-[var(--border-default)] bg-[var(--surface-panel)] px-4 py-3 text-[14.5px] leading-6 text-[var(--text-primary)] outline-none transition-colors placeholder:text-[var(--text-subtle)] focus:border-[var(--border-strong)] disabled:opacity-60";

/** Round play/pause button; `label` names what plays ("option 2"). */
export function PlayButton({
  playing,
  label,
  onClick,
  disabled = false,
  small = false,
}: {
  playing: boolean;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  small?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={`${playing ? "Pause" : "Play"} ${label}`}
      className={cn(
        "flex shrink-0 items-center justify-center rounded-full transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        small ? "h-7 w-7" : "h-8 w-8",
        playing
          ? "bg-[var(--accent-solid)] text-[var(--text-on-accent)]"
          : "bg-[var(--accent-soft)] text-[var(--text-primary)] hover:bg-[var(--surface-raised)]",
      )}
    >
      {playing ? (
        <Pause aria-hidden="true" className="h-3 w-3 fill-current" />
      ) : (
        <Play aria-hidden="true" className="ml-0.5 h-3 w-3 fill-current" />
      )}
    </button>
  );
}

/** The side panel's "Model" row, with "Change" opening the model manager. */
export function VoiceModelRow({
  model,
  modelInfo,
  ready,
  onChange,
  disabled = false,
}: {
  model: string | null;
  modelInfo?: ModelInfo | null;
  ready: boolean;
  onChange: () => void;
  disabled?: boolean;
}) {
  const status = modelInfo?.status;
  const busy = status === "loading" || status === "downloading";
  return (
    <CreationField label="Model">
      <ModelRow
        name={
          model
            ? getModelDisplayName(model, modelInfo?.display_name)
            : "No model selected"
        }
        status={!model ? "missing" : ready ? "ready" : busy ? "busy" : "not_ready"}
        statusLabel={
          !model
            ? "Choose one in Models"
            : ready
              ? "Loaded"
              : status === "downloading"
                ? "Downloading…"
                : status === "loading"
                  ? "Loading…"
                  : "Not loaded"
        }
        actions={
          <ModelRowAction
            onClick={onChange}
            disabled={disabled}
            aria-label="Change model"
          >
            Change
          </ModelRowAction>
        }
      />
    </CreationField>
  );
}

export function FormError({ message }: { message: string | null }) {
  if (!message) {
    return null;
  }
  return (
    <div
      role="alert"
      className="flex items-start gap-2 rounded-[10px] border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2.5 text-[13px] text-[var(--danger-text)]"
    >
      <AlertTriangle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

/** What a creation dialog shows once its voice is saved. */
export function VoiceSavedBody({ name }: { name: string }) {
  return (
    <div className="flex items-start gap-3 px-5 py-5 sm:px-6">
      <CheckCircle2
        aria-hidden="true"
        className="mt-0.5 h-5 w-5 shrink-0 text-[var(--status-positive-text)]"
      />
      <div className="min-w-0">
        <p className="text-[14px] font-medium text-[var(--text-primary)]">
          “{name}” is in My voices.
        </p>
        <p className="mt-1 text-[13px] leading-5 text-[var(--text-muted)]">
          Use it in Text to Speech and Studio.
        </p>
      </div>
    </div>
  );
}
