import { useEffect, useId, useMemo, useRef, useState } from "react";
import {
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Play,
  Square,
  X,
} from "lucide-react";

import type { ModelInfo } from "@/api";
import {
  Sheet,
  SheetContent,
  SheetTitle,
} from "@/components/ui/sheet";
import { Switch } from "@/components/ui/switch";
import { useReadAloud } from "@/features/chat/hooks/useReadAloud";
import {
  VOICE_ENGINES,
  voiceEngineModels,
  voiceEngineSpeakerVariant,
  voiceEngineTotalBytes,
  voiceSpeakerMeta,
  type VoiceEngineModel,
} from "@/features/voice/engine";
import {
  formatBytes,
  isUsableVoiceModelStatus,
  type VoicePageProps,
} from "@/features/voice/realtime/support";
import type {
  VoiceEngineId,
  VoicePreferences,
} from "@/features/voice/useVoicePreferences";
import { cn } from "@/lib/utils";
import type { SpeakerProfile } from "@/types";

export type VoiceSettingsView = "main" | "memories" | "engine";

const VISIBLE_VOICES = 5;
const PROMPT_SAVE_DELAY_MS = 800;

interface VoiceSettingsSheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  view: VoiceSettingsView;
  onViewChange: (view: VoiceSettingsView) => void;
  preferences: VoicePreferences;
  speakers: SpeakerProfile[];
  speaker: SpeakerProfile | null;
  models: ModelInfo[];
  downloadProgress: VoicePageProps["downloadProgress"];
  onDownload: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onError: (message: string) => void;
  /** Changes apply to the next conversation while one is running. */
  isConversationActive: boolean;
}

const sectionTitle = "mb-1.5 text-[13px] font-medium text-[var(--text-muted)]";
const rowLink =
  "flex min-h-[52px] w-full items-center justify-between gap-4 text-left text-sm text-[var(--text-primary)] transition-colors hover:text-[var(--text-secondary)]";

/** Voice settings in one side sheet: no tabs, two sub-views. */
export function VoiceSettingsSheet(props: VoiceSettingsSheetProps) {
  const { open, onOpenChange, view, onViewChange } = props;
  const titles: Record<VoiceSettingsView, string> = {
    main: "Voice settings",
    memories: "Memories",
    engine: "Engine",
  };

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        aria-describedby={undefined}
        className="flex w-full max-w-[420px] flex-col gap-0 border-l border-[var(--border-default)] bg-[var(--bg-surface-1)] p-0 sm:max-w-[420px] [&>button:last-child]:hidden"
      >
        <div className="flex h-16 shrink-0 items-center gap-1 border-b border-[var(--border-muted)] px-3">
          {view !== "main" && (
            <button
              type="button"
              onClick={() => onViewChange("main")}
              aria-label="Back to voice settings"
              className="flex h-10 w-10 items-center justify-center rounded-[10px] text-[var(--text-muted)] transition-colors hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)]"
            >
              <ChevronLeft className="h-[18px] w-[18px]" />
            </button>
          )}
          <SheetTitle
            className={cn(
              "flex-1 text-base font-semibold text-[var(--text-primary)]",
              view === "main" && "pl-3",
            )}
          >
            {titles[view]}
          </SheetTitle>
          <button
            type="button"
            onClick={() => onOpenChange(false)}
            aria-label="Close"
            className="flex h-10 w-10 items-center justify-center rounded-[10px] text-[var(--text-muted)] transition-colors hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)]"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        {view === "main" && <MainView {...props} />}
        {view === "memories" && <MemoriesView preferences={props.preferences} />}
        {view === "engine" && <EngineView {...props} />}
      </SheetContent>
    </Sheet>
  );
}

function engineStatus(rows: VoiceEngineModel[]): {
  label: string;
  loaded: boolean;
} {
  if (rows.some((row) => !row.model || !isUsableVoiceModelStatus(row.model.status))) {
    return { label: "Not downloaded", loaded: false };
  }
  if (rows.every((row) => row.model?.status === "ready")) {
    return { label: "Loaded", loaded: true };
  }
  return { label: "Loads when you start", loaded: false };
}

function MainView({
  onViewChange,
  preferences,
  speakers,
  speaker,
  models,
  onError,
  isConversationActive,
}: VoiceSettingsSheetProps) {
  const voiceGroupId = useId();
  const instructionsId = useId();
  const memoryId = useId();
  const [showAllVoices, setShowAllVoices] = useState(false);
  const preview = useReadAloud(onError);
  const engineRows = voiceEngineModels(preferences.engine, models);
  const status = engineStatus(engineRows);

  const visibleVoices = useMemo(() => {
    if (showAllVoices) return speakers;
    const first = speakers.slice(0, VISIBLE_VOICES);
    if (speaker && !first.some((item) => item.id === speaker.id)) {
      return [speaker, ...first.slice(0, VISIBLE_VOICES - 1)];
    }
    return first;
  }, [showAllVoices, speaker, speakers]);

  const [draft, setDraft] = useState(preferences.savedPrompt);
  // A save of what is being typed (trimmed) leaves the text as typed; a
  // prompt loaded or reset elsewhere replaces it.
  useEffect(() => {
    setDraft((current) =>
      current.trim() === preferences.savedPrompt.trim()
        ? current
        : preferences.savedPrompt,
    );
  }, [preferences.savedPrompt]);

  // Instructions save as you type, and when the sheet closes mid-edit
  // (Escape or the close button unmount the field without a blur).
  const { savePrompt } = preferences;
  const latestDraftRef = useRef(draft);
  const savePromptRef = useRef(savePrompt);
  useEffect(() => {
    latestDraftRef.current = draft;
    savePromptRef.current = savePrompt;
  }, [draft, savePrompt]);
  useEffect(() => {
    const timer = window.setTimeout(() => void savePrompt(draft), PROMPT_SAVE_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [draft, savePrompt]);
  useEffect(
    () => () => void savePromptRef.current(latestDraftRef.current),
    [],
  );

  const promptHint =
    preferences.promptSaveState === "saving"
      ? "Saving…"
      : preferences.promptSaveState === "error"
        ? "Couldn't save"
        : isConversationActive
          ? "Applies to your next conversation"
          : preferences.promptSaveState === "saved"
            ? "Saved · applies to your next conversation"
            : "Applies to your next conversation";

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-7 overflow-y-auto px-6 pb-8 pt-6">
      <section aria-labelledby={voiceGroupId} className="flex flex-col">
        <h3 id={voiceGroupId} className={sectionTitle}>
          Voice
        </h3>
        {speakers.length === 0 ? (
          <p className="text-sm text-[var(--text-muted)]">
            This engine has no built-in voices.
          </p>
        ) : (
          <div role="radiogroup" aria-labelledby={voiceGroupId} className="flex flex-col gap-0.5">
            {visibleVoices.map((item) => {
              const checked = item.id === speaker?.id;
              const previewing = preview.state?.messageId === item.id;
              return (
                <div
                  key={item.id}
                  className={cn(
                    "flex items-center gap-2 rounded-[10px] pl-2.5 pr-2",
                    checked && "bg-[var(--accent-soft)]",
                  )}
                >
                  <button
                    type="button"
                    role="radio"
                    aria-checked={checked}
                    onClick={() => preferences.setSpeaker(item.id)}
                    className="flex h-[52px] min-w-0 flex-1 items-center gap-3 text-left"
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
                      <span className="text-sm font-medium text-[var(--text-primary)]">
                        {item.name}
                      </span>
                      <span className="truncate text-xs text-[var(--text-muted)]">
                        {voiceSpeakerMeta(item)}
                      </span>
                    </span>
                  </button>
                  <button
                    type="button"
                    aria-label={previewing ? `Stop ${item.name}` : `Preview ${item.name}`}
                    onClick={() =>
                      previewing
                        ? preview.stop()
                        : preview.start(
                            item.id,
                            `Hi, I'm ${item.name}. This is how I sound.`,
                            {
                              modelId: voiceEngineSpeakerVariant(preferences.engine),
                              speaker: item.id,
                            },
                          )
                    }
                    className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-[var(--border-default)] text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-surface-2)]"
                  >
                    {previewing && preview.state?.status === "loading" ? (
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
        {speakers.length > VISIBLE_VOICES && (
          <button
            type="button"
            onClick={() => setShowAllVoices((current) => !current)}
            aria-expanded={showAllVoices}
            className="flex h-10 items-center gap-1.5 px-2.5 text-[13px] text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)]"
          >
            {showAllVoices ? "Show fewer voices" : `Show all ${speakers.length} voices`}
            <ChevronDown
              className={cn("h-3.5 w-3.5 transition-transform", showAllVoices && "rotate-180")}
            />
          </button>
        )}
      </section>

      <section className="flex flex-col gap-2.5">
        <label htmlFor={instructionsId} className={sectionTitle}>
          Instructions
        </label>
        <textarea
          id={instructionsId}
          rows={5}
          value={draft}
          disabled={preferences.isProfileLoading}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={() => void preferences.savePrompt(draft)}
          className="w-full resize-y rounded-xl border border-[var(--border-default)] bg-[var(--bg-surface-2)] px-3.5 py-3 text-sm leading-relaxed text-[var(--text-primary)] outline-none transition-colors placeholder:text-[var(--text-subtle)] focus:border-[var(--border-strong)] disabled:opacity-60"
        />
        <div className="flex items-center justify-between gap-3 text-xs text-[var(--text-subtle)]">
          <span className="flex items-center gap-1.5">
            {preferences.promptSaveState === "saved" && !isConversationActive && (
              <Check className="h-[13px] w-[13px]" strokeWidth={2.5} />
            )}
            {promptHint}
          </span>
          {(preferences.isPromptCustom || draft.trim() !== preferences.defaultPrompt.trim()) && (
            <button
              type="button"
              onClick={() => {
                setDraft(preferences.defaultPrompt);
                void preferences.savePrompt(preferences.defaultPrompt);
              }}
              className="py-1.5 text-[var(--text-muted)] underline underline-offset-[3px] hover:text-[var(--text-primary)]"
            >
              Reset to default
            </button>
          )}
        </div>
      </section>

      <section className="flex flex-col">
        <h3 className={sectionTitle}>Memory</h3>
        <div className="flex min-h-[52px] items-center justify-between gap-4">
          <label htmlFor={memoryId} className="flex flex-col gap-0.5">
            <span className="text-sm font-medium text-[var(--text-primary)]">
              Remember what I tell you
            </span>
            <span className="text-xs text-[var(--text-muted)]">
              Keeps preferences and facts between conversations
            </span>
          </label>
          <Switch
            id={memoryId}
            checked={preferences.memoryEnabled}
            disabled={preferences.isProfileLoading}
            onCheckedChange={(checked) => void preferences.setMemoryEnabled(checked)}
          />
        </div>
        <button
          type="button"
          onClick={() => onViewChange("memories")}
          className={cn(rowLink, "border-t border-[var(--border-muted)]")}
        >
          <span>Manage memories</span>
          <span className="flex items-center gap-1.5 text-[var(--text-muted)]">
            {preferences.memories?.length ?? ""}
            <ChevronRight className="h-4 w-4" />
          </span>
        </button>
      </section>

      <section className="flex flex-col">
        <h3 className={sectionTitle}>Engine</h3>
        <button type="button" onClick={() => onViewChange("engine")} className={rowLink}>
          <span className="flex min-w-0 flex-col gap-0.5">
            <span className="font-medium">{VOICE_ENGINES[preferences.engine].label}</span>
            <span className="truncate text-xs text-[var(--text-muted)]">
              {engineRows.map((row) => row.label).join(" · ")}
            </span>
          </span>
          <span className="flex shrink-0 items-center gap-2 text-[13px] text-[var(--text-muted)]">
            <span
              aria-hidden="true"
              className={cn(
                "h-[7px] w-[7px] rounded-full",
                status.loaded
                  ? "bg-[var(--status-positive-solid)]"
                  : "border-[1.5px] border-[var(--border-strong)]",
              )}
            />
            {status.label}
            <ChevronRight className="h-4 w-4" />
          </span>
        </button>
      </section>
    </div>
  );
}

function formatMemoryDate(timestamp: number): string {
  const ms = timestamp > 1e12 ? timestamp : timestamp * 1000;
  return new Date(ms).toLocaleDateString([], { month: "short", day: "numeric" });
}

function MemoriesView({ preferences }: { preferences: VoicePreferences }) {
  const { memories, loadMemories } = preferences;
  const [confirmingForgetAll, setConfirmingForgetAll] = useState(false);
  useEffect(() => {
    void loadMemories();
  }, [loadMemories]);

  return (
    <>
      <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-6 pb-6 pt-4">
        <p className="text-[13px] leading-relaxed text-[var(--text-muted)]">
          Izwi remembers lasting preferences and facts from your voice
          conversations. Forget anything you don’t want kept.
        </p>
        {memories === null ? (
          <p className="text-sm text-[var(--text-muted)]">Loading…</p>
        ) : memories.length === 0 ? (
          <p className="border-t border-[var(--border-muted)] pt-4 text-sm text-[var(--text-muted)]">
            Nothing remembered yet.
          </p>
        ) : (
          <ul>
            {memories.map((memory) => (
              <li
                key={memory.id}
                className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 border-t border-[var(--border-muted)] py-3"
              >
                <div className="flex min-w-0 flex-col gap-0.5">
                  <span className="text-sm leading-snug text-[var(--text-primary)]">
                    {memory.summary}
                  </span>
                  <span className="text-xs text-[var(--text-subtle)]">
                    {formatMemoryDate(memory.updated_at)}
                  </span>
                </div>
                <button
                  type="button"
                  aria-label={`Forget: ${memory.summary}`}
                  disabled={preferences.isMemoryBusy}
                  onClick={() => void preferences.forgetMemory(memory.id)}
                  className="flex h-9 w-9 items-center justify-center rounded-lg text-[var(--text-muted)] transition-colors hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)] disabled:opacity-50"
                >
                  <X className="h-[15px] w-[15px]" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
      {memories && memories.length > 0 && (
        <div className="flex shrink-0 items-center gap-3 border-t border-[var(--border-muted)] px-6 py-3.5">
          {confirmingForgetAll ? (
            <>
              <span className="flex-1 text-sm text-[var(--text-primary)]">
                Forget all {memories.length} memories?
              </span>
              <button
                type="button"
                onClick={() => setConfirmingForgetAll(false)}
                className="h-10 rounded-full px-3 text-sm text-[var(--text-muted)] hover:text-[var(--text-primary)]"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={preferences.isMemoryBusy}
                onClick={() => {
                  setConfirmingForgetAll(false);
                  void preferences.forgetAllMemories();
                }}
                className="h-10 rounded-full bg-red-600 px-4 text-sm font-medium text-white hover:bg-red-700 disabled:opacity-50"
              >
                Forget all
              </button>
            </>
          ) : (
            <button
              type="button"
              onClick={() => setConfirmingForgetAll(true)}
              className="h-10 px-1 text-sm text-[var(--danger-text)] hover:underline"
            >
              Forget all memories…
            </button>
          )}
        </div>
      )}
    </>
  );
}

function modelStatusLabel(
  model: ModelInfo | null,
  progress: VoicePageProps["downloadProgress"],
): string {
  if (!model) return "Unavailable";
  switch (model.status) {
    case "ready":
      return "Loaded";
    case "loading":
      return "Loading";
    case "downloaded":
      return "Downloaded";
    case "downloading":
      return `Downloading · ${Math.round(progress[model.variant]?.percent ?? model.download_progress ?? 0)}%`;
    case "error":
      return "Failed";
    default:
      return "Not downloaded";
  }
}

function EngineView({
  preferences,
  models,
  downloadProgress,
  onDownload,
  onLoad,
  onUnload,
  isConversationActive,
}: VoiceSettingsSheetProps) {
  const groupId = useId();
  const engines: VoiceEngineId[] = ["standard", "compact"];
  const rows = voiceEngineModels(preferences.engine, models);
  const missing = rows.filter(
    (row) =>
      row.model &&
      (row.model.status === "not_downloaded" || row.model.status === "error"),
  );
  const downloaded = rows.filter((row) => row.model?.status === "downloaded");
  const loaded = rows.filter((row) => row.model?.status === "ready");

  let action: { label: string; primary: boolean; run: () => void } | null = null;
  let note = "Models load when a conversation starts.";
  if (missing.length > 0) {
    action = {
      label: "Download",
      primary: true,
      run: () => missing.forEach((row) => onDownload(row.variant)),
    };
    note = "Downloads the models this engine still needs.";
  } else if (downloaded.length > 0) {
    action = {
      label: "Load now",
      primary: false,
      run: () => downloaded.forEach((row) => onLoad(row.variant)),
    };
  } else if (loaded.length > 0) {
    action = {
      label: "Unload",
      primary: false,
      run: () => loaded.forEach((row) => onUnload(row.variant)),
    };
    note = "Unloading frees memory until your next conversation.";
  }
  if (isConversationActive) {
    note = "Changes apply to your next conversation.";
  }

  return (
    <>
      <div className="flex min-h-0 flex-1 flex-col gap-7 overflow-y-auto px-6 pb-6 pt-5">
        <div role="radiogroup" aria-labelledby={groupId} className="flex flex-col gap-2.5">
          <span id={groupId} className="sr-only">
            Engine
          </span>
          {engines.map((id) => {
            const checked = preferences.engine === id;
            const bytes = voiceEngineTotalBytes(voiceEngineModels(id, models));
            return (
              <button
                key={id}
                type="button"
                role="radio"
                aria-checked={checked}
                onClick={() => preferences.setEngine(id)}
                className={cn(
                  "flex w-full items-start gap-3 rounded-xl border px-4 py-3.5 text-left transition-colors",
                  checked
                    ? "border-[var(--border-strong)] bg-[var(--accent-soft)]"
                    : "border-[var(--border-muted)] hover:border-[var(--border-default)]",
                )}
              >
                <span
                  aria-hidden="true"
                  className={cn(
                    "mt-0.5 h-4 w-4 shrink-0 rounded-full",
                    checked
                      ? "border-[5px] border-[var(--accent-solid)]"
                      : "border-[1.5px] border-[var(--border-strong)]",
                  )}
                />
                <span className="flex min-w-0 flex-1 flex-col gap-1">
                  <span className="flex justify-between gap-3">
                    <span className="text-sm font-medium text-[var(--text-primary)]">
                      {VOICE_ENGINES[id].label}
                    </span>
                    {bytes > 0 && (
                      <span className="whitespace-nowrap text-[13px] text-[var(--text-muted)]">
                        {formatBytes(bytes)}
                      </span>
                    )}
                  </span>
                  <span className="text-[13px] leading-snug text-[var(--text-muted)]">
                    {VOICE_ENGINES[id].description}
                  </span>
                </span>
              </button>
            );
          })}
        </div>

        <section className="flex flex-col">
          <h3 className={sectionTitle}>Models</h3>
          <ul>
            {rows.map((row) => (
              <li
                key={row.variant}
                className="grid min-h-[52px] grid-cols-[minmax(0,1fr)_auto] items-center gap-3 border-t border-[var(--border-muted)]"
              >
                <span className="flex min-w-0 flex-col gap-0.5">
                  <span className="truncate text-sm font-medium text-[var(--text-primary)]">
                    {row.label}
                  </span>
                  <span className="text-xs text-[var(--text-muted)]">{row.role}</span>
                </span>
                <span className="flex items-center gap-2 whitespace-nowrap text-[13px] text-[var(--text-muted)]">
                  <span
                    aria-hidden="true"
                    className={cn(
                      "h-[7px] w-[7px] rounded-full",
                      row.model?.status === "ready"
                        ? "bg-[var(--status-positive-solid)]"
                        : "border-[1.5px] border-[var(--border-strong)]",
                    )}
                  />
                  {modelStatusLabel(row.model, downloadProgress)}
                </span>
              </li>
            ))}
          </ul>
        </section>
      </div>
      <div className="flex shrink-0 items-center justify-between gap-3 border-t border-[var(--border-muted)] px-6 py-3.5">
        <span className="text-xs text-[var(--text-subtle)]">{note}</span>
        {action && (
          <button
            type="button"
            onClick={action.run}
            className={cn(
              "h-10 shrink-0 rounded-full px-4 text-sm",
              action.primary
                ? "bg-[var(--accent-solid)] font-medium text-[var(--text-on-accent)] hover:opacity-90"
                : "border border-[var(--border-default)] text-[var(--text-primary)] hover:bg-[var(--bg-surface-2)]",
            )}
          >
            {action.label}
          </button>
        )}
      </div>
    </>
  );
}
