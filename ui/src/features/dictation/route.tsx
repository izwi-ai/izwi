import { useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router";
import { Check, Copy, Loader2, RotateCcw } from "lucide-react";

import type { ModelInfo } from "@/api";
import { PageHeader, PageShell } from "@/components/PageShell";
import { SettingsRow, SettingsSection } from "@/components/SettingsSection";
import { Button } from "@/components/ui/button";
import { PageLoading } from "@/components/ui/page-state";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { StatePanel } from "@/components/ui/state-panel";
import { Switch } from "@/components/ui/switch";
import { dictationApi } from "@/features/dictation/api";
import { HotkeyRecorder } from "@/features/dictation/components/HotkeyRecorder";
import {
  SetupChecklist,
  keyboardPermissionSettled,
  speechModelSettled,
} from "@/features/dictation/components/SetupChecklist";
import {
  ReplacementsEditor,
  VocabularyInput,
} from "@/features/dictation/components/TextRules";
import {
  activityText,
  readyText,
  resultText,
} from "@/features/dictation/statusText";
import {
  DEFAULT_DICTATION_MODEL,
  type DictationHistoryEntry,
  type DictationMode,
  type DictationResultStatus,
  type DictationSnapshot,
  type InsertMethod,
} from "@/features/dictation/types";
import {
  useDictation,
  type UseDictationResult,
} from "@/features/dictation/useDictation";
import { getModelDisplayName } from "@/features/models/catalog/modelMetadata";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";
import { modelTaskGroupId } from "@/features/models/taskGroups";
import { LANGUAGE_OPTIONS } from "@/features/transcription/playground/support";

export interface DictationRouteProps {
  models: ModelInfo[];
  downloadProgress: ModelDownloadProgressMap;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
}

const PAGE_DESCRIPTION =
  "Hold a shortcut in any app, speak, and Izwi writes what you said at the cursor.";

/** Set once a dictation has been inserted, so setup stays folded afterwards. */
const TRIED_STORAGE_KEY = "izwi.dictation.tried";

/** Select value for "no choice": the default model, auto language, and so on. */
const UNSET = "__unset__";

const SELECT_TRIGGER_CLASS =
  "h-9 w-64 max-w-full rounded-[10px] border-(--border-default) bg-(--surface-panel) text-[13.5px]";

const LINK_CLASS =
  "text-(--text-primary) underline underline-offset-2 hover:no-underline";

function readTried(): boolean {
  try {
    return window.localStorage.getItem(TRIED_STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

function rememberTried() {
  try {
    window.localStorage.setItem(TRIED_STORAGE_KEY, "1");
  } catch {
    // Setup simply shows again next time.
  }
}

function isInstalled(model: ModelInfo): boolean {
  return (
    model.status === "downloaded" ||
    model.status === "loading" ||
    model.status === "ready"
  );
}

function FieldError({ message }: { message?: string | null }) {
  return message ? (
    <p role="alert" className="mt-1.5 text-[12.5px] text-(--danger-text)">
      {message}
    </p>
  ) : null;
}

function SwitchRow({
  title,
  description,
  checked,
  onCheckedChange,
  error,
  disabled,
}: {
  title: string;
  description?: ReactNode;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  error?: string;
  disabled?: boolean;
}) {
  return (
    <SettingsRow
      title={title}
      description={
        <>
          {description}
          <FieldError message={error} />
        </>
      }
      action={
        <Switch
          checked={checked}
          disabled={disabled}
          onCheckedChange={onCheckedChange}
          aria-label={title}
        />
      }
    />
  );
}

function SelectRow({
  title,
  description,
  value,
  onValueChange,
  options,
  error,
  footer,
}: {
  title: string;
  description?: ReactNode;
  value: string;
  onValueChange: (value: string) => void;
  options: { value: string; label: string }[];
  error?: string;
  footer?: ReactNode;
}) {
  return (
    <SettingsRow
      title={title}
      description={
        <>
          {description}
          {footer}
          <FieldError message={error} />
        </>
      }
      action={
        <Select value={value} onValueChange={onValueChange}>
          <SelectTrigger aria-label={title} className={SELECT_TRIGGER_CLASS}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {options.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      }
    />
  );
}

function statusLine(
  snapshot: DictationSnapshot,
  lastResult: DictationResultStatus | null,
): { text: string; tone: "neutral" | "danger"; busy: boolean } {
  const { settings, status, platform } = snapshot;
  if (!settings.enabled) {
    return {
      text: "Dictation is off. Turn it on to use your shortcut.",
      tone: "neutral",
      busy: false,
    };
  }
  const activity = activityText(status);
  if (activity) {
    return { text: activity, tone: "neutral", busy: true };
  }
  if (lastResult) {
    const failed =
      lastResult.state === "failed" ||
      (lastResult.state === "inserted" && lastResult.outcome.kind === "failed");
    return {
      text: resultText(lastResult, platform.os),
      tone: failed ? "danger" : "neutral",
      busy: false,
    };
  }
  if (snapshot.hotkey_error) {
    return { text: snapshot.hotkey_error, tone: "danger", busy: false };
  }
  if (!speechModelSettled(snapshot)) {
    return {
      text: "Finish setup below to start dictating.",
      tone: "neutral",
      busy: false,
    };
  }
  return { text: readyText(settings, platform.os), tone: "neutral", busy: false };
}

function formatHistoryTime(atMs: number): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(atMs));
}

const HISTORY_OUTCOME_NOTES: Partial<
  Record<DictationHistoryEntry["outcome"], string>
> = {
  clipboard: "Copied",
  failed: "Not inserted",
};

function HistoryCopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) {
      return;
    }
    const timer = window.setTimeout(() => setCopied(false), 2000);
    return () => window.clearTimeout(timer);
  }, [copied]);

  const Icon = copied ? Check : Copy;
  return (
    <button
      type="button"
      aria-label={copied ? "Copied" : "Copy dictation"}
      onClick={() =>
        void dictationApi
          .copyText(text)
          .then(() => setCopied(true))
          .catch(() => undefined)
      }
      className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-(--text-subtle) transition-colors hover:bg-(--bg-surface-2) hover:text-(--text-primary) focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-(--border-strong)"
    >
      <Icon className="h-4 w-4" />
    </button>
  );
}

function DictationSettingsView({
  dictation,
  snapshot,
  models,
  downloadProgress,
  onDownload,
  onCancelDownload,
}: DictationRouteProps & {
  dictation: UseDictationResult;
  snapshot: DictationSnapshot;
}) {
  const { settings, platform } = snapshot;
  const { update, errors, lastResult, history } = dictation;
  const [triedBefore] = useState(readTried);
  const [setupOpenChoice, setSetupOpenChoice] = useState<boolean | null>(null);
  const [clearError, setClearError] = useState<string | null>(null);

  useEffect(() => {
    if (lastResult?.state === "inserted") {
      rememberTried();
    }
  }, [lastResult]);

  const setupDone =
    speechModelSettled(snapshot) && keyboardPermissionSettled(snapshot);
  // Folded once everything is in place and dictation has been used; setup
  // finished during this visit stays open so the first try is right there.
  const setupFolded = setupDone && triedBefore;
  const setupOpen = setupOpenChoice ?? !setupFolded;

  const line = statusLine(snapshot, lastResult);
  const elevated =
    lastResult?.state === "inserted" &&
    lastResult.outcome.kind === "clipboard" &&
    lastResult.outcome.reason === "elevated";

  const speechModels = models.filter(
    (model) =>
      modelTaskGroupId(model) === "asr" &&
      isInstalled(model) &&
      model.variant !== DEFAULT_DICTATION_MODEL,
  );
  const chatModels = models.filter(
    (model) => modelTaskGroupId(model) === "chat" && isInstalled(model),
  );
  const modelOptions = (installed: ModelInfo[], chosen: string | null) => {
    const options = installed.map((model) => ({
      value: model.variant,
      label: getModelDisplayName(model.variant, model.display_name),
    }));
    // A saved choice that is no longer installed still shows as the choice.
    if (chosen && !options.some((option) => option.value === chosen)) {
      options.push({
        value: chosen,
        label: `${getModelDisplayName(chosen)} (not installed)`,
      });
    }
    return options;
  };
  const speechModelChoice =
    settings.model && settings.model !== DEFAULT_DICTATION_MODEL
      ? settings.model
      : null;

  const microphoneOptions = snapshot.microphones.map((microphone) => ({
    value: microphone.name,
    label: microphone.name,
  }));
  if (
    settings.microphone &&
    !microphoneOptions.some((option) => option.value === settings.microphone)
  ) {
    microphoneOptions.push({
      value: settings.microphone,
      label: `${settings.microphone} (not connected)`,
    });
  }

  const knownLanguages: readonly string[] = LANGUAGE_OPTIONS;
  const languageOptions =
    settings.language && !knownLanguages.includes(settings.language)
      ? [settings.language, ...knownLanguages]
      : knownLanguages;

  const modelsLink = (
    <>
      {" "}
      <Link to="/models" className={LINK_CLASS}>
        Get more models
      </Link>
    </>
  );

  return (
    <PageShell>
      <PageHeader
        title="Dictation"
        description={PAGE_DESCRIPTION}
        actions={
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-(--text-secondary)">
              {settings.enabled ? "On" : "Off"}
            </span>
            <Switch
              checked={settings.enabled}
              onCheckedChange={(enabled) => void update({ enabled })}
              aria-label="Dictation"
            />
          </div>
        }
      />

      <div className="max-w-3xl space-y-8">
        <div>
          <p
            role="status"
            className={
              line.tone === "danger"
                ? "flex items-center gap-2 text-[13.5px] text-(--danger-text)"
                : "flex items-center gap-2 text-[13.5px] text-(--text-secondary)"
            }
          >
            {line.busy ? (
              <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />
            ) : null}
            <span className="min-w-0 truncate">{line.text}</span>
          </p>
          <FieldError message={errors.enabled} />
        </div>

        {platform.wayland ? (
          <StatePanel
            tone="info"
            title="Wayland"
            description={
              <>
                Apps can't always listen for a shortcut on Wayland. If yours
                does nothing, bind a key in your desktop's keyboard settings to
                run <code>izwi-desktop --dictation toggle</code>. Pasting needs{" "}
                <code>wtype</code>; without it your words are left on the
                clipboard.
              </>
            }
          />
        ) : null}
        {snapshot.server_is_remote ? (
          <StatePanel
            tone="info"
            title="Remote server"
            description="Izwi is connected to a server on another computer, so the audio you dictate is sent to that server to be transcribed."
          />
        ) : null}
        {elevated ? (
          <StatePanel
            tone="warning"
            title="Administrator window"
            description="Windows doesn't let apps type into windows that run as administrator. Your words are on the clipboard: paste them with Ctrl+V."
          />
        ) : null}

        <SettingsSection
          id="dictation-setup"
          title="Setup"
          description={
            setupDone ? (
              <span className="inline-flex items-center gap-2">
                Everything is in place.
                <button
                  type="button"
                  aria-expanded={setupOpen}
                  aria-controls="dictation-setup-steps"
                  onClick={() => setSetupOpenChoice(!setupOpen)}
                  className={LINK_CLASS}
                >
                  {setupOpen ? "Hide" : "Show"}
                </button>
              </span>
            ) : (
              "What dictation needs before it can write for you."
            )
          }
        >
          <div id="dictation-setup-steps">
            {setupOpen ? (
              <SetupChecklist
                snapshot={snapshot}
                partial={dictation.partial}
                models={models}
                downloadProgress={downloadProgress}
                onDownload={onDownload}
                onCancelDownload={onCancelDownload}
                onRefresh={dictation.refresh}
                onEnable={() => void update({ enabled: true })}
              />
            ) : (
              <SettingsRow description="The speech model is ready and Izwi can write into other apps." />
            )}
          </div>
        </SettingsSection>

        <SettingsSection id="dictation-shortcut" title="Shortcut">
          <SettingsRow
            title="Dictation shortcut"
            description={
              <>
                Works in any app while Izwi is running.
                <FieldError message={errors.hotkey ?? snapshot.hotkey_error} />
              </>
            }
            action={
              <div className="flex flex-wrap items-start gap-2 md:justify-end">
                {platform.fn_key ? (
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    disabled={settings.hotkey === "Fn"}
                    onClick={() => void update({ hotkey: "Fn" })}
                  >
                    Use Fn
                  </Button>
                ) : null}
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  disabled={settings.hotkey === platform.default_hotkey}
                  onClick={() => void update({ hotkey: platform.default_hotkey })}
                >
                  <RotateCcw />
                  Reset
                </Button>
                <HotkeyRecorder
                  label="Dictation shortcut"
                  value={settings.hotkey}
                  os={platform.os}
                  modifierOnly={platform.modifier_only_hotkeys}
                  onChange={(hotkey) => void update({ hotkey })}
                />
              </div>
            }
          />
          <SelectRow
            title="How it works"
            value={settings.mode}
            onValueChange={(mode) =>
              void update({ mode: mode as DictationMode })
            }
            options={[
              { value: "hold", label: "Hold to talk" },
              { value: "toggle", label: "Press to start and stop" },
            ]}
            description={
              settings.mode === "hold"
                ? "Hold the shortcut while you speak and let go when you're done."
                : "Press the shortcut to start and press it again to finish."
            }
            error={errors.mode}
          />
          {settings.mode === "hold" ? (
            <SwitchRow
              title="Double-tap for hands-free"
              description="Press the shortcut twice quickly to keep recording until you press it again."
              checked={settings.double_tap_lock}
              onCheckedChange={(double_tap_lock) =>
                void update({ double_tap_lock })
              }
              error={errors.double_tap_lock}
            />
          ) : null}
          <SettingsRow
            title="Paste last dictation"
            description={
              <>
                Optional. Inserts your most recent dictation again.
                <FieldError
                  message={
                    errors.paste_last_hotkey ?? snapshot.paste_last_hotkey_error
                  }
                />
              </>
            }
            action={
              <HotkeyRecorder
                label="Paste last dictation shortcut"
                value={settings.paste_last_hotkey}
                os={platform.os}
                modifierOnly={false}
                onChange={(paste_last_hotkey) =>
                  void update({ paste_last_hotkey })
                }
                onClear={() => void update({ paste_last_hotkey: null })}
              />
            }
          />
        </SettingsSection>

        <SettingsSection id="dictation-speech" title="Speech">
          <SelectRow
            title="Speech model"
            value={speechModelChoice ?? UNSET}
            onValueChange={(value) =>
              void update({ model: value === UNSET ? null : value })
            }
            options={[
              { value: UNSET, label: "Parakeet TDT 0.6B v3 (recommended)" },
              ...modelOptions(speechModels, speechModelChoice),
            ]}
            description="Installed speech-to-text models."
            footer={modelsLink}
            error={errors.model}
          />
          <SelectRow
            title="Language"
            value={settings.language ?? UNSET}
            onValueChange={(value) =>
              void update({ language: value === UNSET ? null : value })
            }
            options={[
              { value: UNSET, label: "Auto-detect" },
              ...languageOptions.map((language) => ({
                value: language,
                label: language,
              })),
            ]}
            description="The language you speak."
            error={errors.language}
          />
          <SelectRow
            title="Microphone"
            value={settings.microphone ?? UNSET}
            onValueChange={(value) =>
              void update({ microphone: value === UNSET ? null : value })
            }
            options={[
              { value: UNSET, label: "System default" },
              ...microphoneOptions,
            ]}
            footer={
              platform.os === "macos" ? (
                <button
                  type="button"
                  onClick={() =>
                    void dictationApi
                      .openSystemSettings("microphone")
                      .catch(() => undefined)
                  }
                  className={LINK_CLASS}
                >
                  Open microphone privacy settings
                </button>
              ) : null
            }
            error={errors.microphone}
          />
          <SwitchRow
            title="Keep microphone ready"
            description="Recording starts instantly, so your first word is never clipped. The system's microphone indicator stays on while dictation is on."
            checked={settings.keep_microphone_ready}
            onCheckedChange={(keep_microphone_ready) =>
              void update({ keep_microphone_ready })
            }
            error={errors.keep_microphone_ready}
          />
        </SettingsSection>

        <SettingsSection id="dictation-text" title="Text">
          <SelectRow
            title="Insert text by"
            value={settings.insert_method}
            onValueChange={(method) =>
              void update({ insert_method: method as InsertMethod })
            }
            options={[
              { value: "paste", label: "Pasting" },
              { value: "type", label: "Typing" },
              { value: "clipboard", label: "Copying to the clipboard only" },
            ]}
            description={
              settings.insert_method === "paste"
                ? "Fastest. Your clipboard is put back afterwards."
                : settings.insert_method === "type"
                  ? "Types each character. Slower, and leaves the clipboard alone."
                  : "Nothing is inserted. Paste the text yourself."
            }
            error={errors.insert_method}
          />
          <SwitchRow
            title="Add a space after"
            description="Ends each dictation with a space, ready for the next one."
            checked={settings.trailing_space}
            onCheckedChange={(trailing_space) => void update({ trailing_space })}
            error={errors.trailing_space}
          />
          <SwitchRow
            title="Remove filler words"
            description="Drops words like “um” and “uh”."
            checked={settings.remove_fillers}
            onCheckedChange={(remove_fillers) => void update({ remove_fillers })}
            error={errors.remove_fillers}
          />
          <SettingsRow
            title="Vocabulary"
            description={
              <>
                Names and terms to spell the way you write them.
                <FieldError message={errors.vocabulary} />
              </>
            }
          >
            <VocabularyInput
              terms={settings.vocabulary}
              onChange={(vocabulary) => void update({ vocabulary })}
            />
          </SettingsRow>
          <SettingsRow
            title="Replacements"
            description={
              <>
                Swap a spoken phrase for other text, such as “my email” for
                your address.
                <FieldError message={errors.replacements} />
              </>
            }
          >
            <ReplacementsEditor
              replacements={settings.replacements}
              onChange={(replacements) => void update({ replacements })}
            />
          </SettingsRow>
          <SwitchRow
            title="Clean up with AI"
            description="A chat model fixes punctuation and slips of the tongue. It adds a short delay; if it takes too long, your words are inserted as spoken."
            checked={settings.cleanup.enabled}
            onCheckedChange={(enabled) => void update({ cleanup: { enabled } })}
            error={errors.cleanup}
          />
          {settings.cleanup.enabled ? (
            <SelectRow
              title="Clean-up model"
              value={settings.cleanup.model ?? UNSET}
              onValueChange={(value) =>
                void update({
                  cleanup: { model: value === UNSET ? null : value },
                })
              }
              options={[
                { value: UNSET, label: "Automatic" },
                ...modelOptions(chatModels, settings.cleanup.model),
              ]}
              description="Installed chat models. Automatic lets Izwi choose."
              footer={modelsLink}
            />
          ) : null}
        </SettingsSection>

        <SettingsSection id="dictation-feedback" title="Feedback">
          <SwitchRow
            title="Show indicator"
            description="A small pill on screen while Izwi listens and transcribes."
            checked={settings.show_indicator}
            onCheckedChange={(show_indicator) => void update({ show_indicator })}
            error={errors.show_indicator}
          />
          <SwitchRow
            title="Sounds"
            description="Plays a sound when recording starts and stops."
            checked={settings.sounds}
            onCheckedChange={(sounds) => void update({ sounds })}
            error={errors.sounds}
          />
          <SwitchRow
            title="Mute other audio while recording"
            description="Silences music and videos so they aren't picked up."
            checked={settings.mute_while_recording}
            onCheckedChange={(mute_while_recording) =>
              void update({ mute_while_recording })
            }
            error={errors.mute_while_recording}
          />
          <SwitchRow
            title="Live preview"
            description="Shows your words in the indicator as you speak, with models that support it."
            checked={settings.live_preview}
            onCheckedChange={(live_preview) => void update({ live_preview })}
            error={errors.live_preview}
          />
        </SettingsSection>

        <SettingsSection id="dictation-history" title="History">
          <SwitchRow
            title="Keep history"
            description="Saves recent dictations on this computer so you can copy them again."
            checked={settings.keep_history}
            onCheckedChange={(keep_history) => void update({ keep_history })}
            error={errors.keep_history}
          />
          {history.length > 0 ? (
            <SettingsRow
              title="Recent dictations"
              description={<FieldError message={clearError} />}
              action={
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    setClearError(null);
                    void dictation
                      .clearHistory()
                      .catch(() =>
                        setClearError("The history could not be cleared."),
                      );
                  }}
                >
                  Clear
                </Button>
              }
            >
              <ul aria-label="Recent dictations" className="space-y-3">
                {history.slice(0, 50).map((entry) => (
                  <li key={entry.id} className="flex items-start gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="text-[12.5px] text-(--text-subtle)">
                        {[
                          formatHistoryTime(entry.at_ms),
                          entry.app,
                          HISTORY_OUTCOME_NOTES[entry.outcome],
                        ]
                          .filter(Boolean)
                          .join(" · ")}
                      </div>
                      <p className="mt-0.5 whitespace-pre-wrap break-words text-[13.5px] text-(--text-primary)">
                        {entry.text}
                      </p>
                    </div>
                    <HistoryCopyButton text={entry.text} />
                  </li>
                ))}
              </ul>
            </SettingsRow>
          ) : (
            <SettingsRow
              description={
                settings.keep_history
                  ? "Nothing dictated yet."
                  : "History is off. Nothing is saved."
              }
            />
          )}
        </SettingsSection>
      </div>
    </PageShell>
  );
}

export function DictationPage(props: DictationRouteProps) {
  const dictation = useDictation();

  if (!dictation.available) {
    return (
      <PageShell>
        <PageHeader title="Dictation" description={PAGE_DESCRIPTION} />
        <StatePanel
          className="max-w-3xl"
          title="Dictation is part of the desktop app"
          description="It needs a system-wide shortcut and the ability to write into other apps, which a browser can't do. Open Izwi's desktop app to set it up."
        />
      </PageShell>
    );
  }

  if (!dictation.snapshot) {
    if (!dictation.loadError) {
      return <PageLoading />;
    }
    return (
      <PageShell>
        <PageHeader title="Dictation" description={PAGE_DESCRIPTION} />
        <StatePanel
          className="max-w-3xl"
          tone="danger"
          title="Dictation could not be loaded"
          description={dictation.loadError}
          actions={
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => void dictation.refresh()}
            >
              Try again
            </Button>
          }
        />
      </PageShell>
    );
  }

  return (
    <DictationSettingsView
      {...props}
      dictation={dictation}
      snapshot={dictation.snapshot}
    />
  );
}
