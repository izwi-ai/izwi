import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { Mic, SlidersHorizontal } from "lucide-react";

import { PageShell } from "@/components/PageShell";
import {
  isVoiceEngineReady,
  resolveCompactVoiceModel,
  resolveVoiceSpeaker,
  voiceEngineDownloadBytes,
  voiceEngineModels,
  voiceEngineSpeakers,
  voiceSessionEngine,
} from "@/features/voice/engine";
import { VoiceControls } from "@/features/voice/components/VoiceControls";
import { VoiceFailure } from "@/features/voice/components/VoiceFailure";
import { VoiceOrb } from "@/features/voice/components/VoiceOrb";
import {
  VoiceSettingsSheet,
  type VoiceSettingsView,
} from "@/features/voice/components/VoiceSettingsSheet";
import { VoiceSetup } from "@/features/voice/components/VoiceSetup";
import { VoiceTranscript } from "@/features/voice/components/VoiceTranscript";
import {
  formatModelVariantLabel,
  type RuntimeStatus,
  type VoicePageProps,
} from "@/features/voice/realtime/support";
import { useVoicePreferences } from "@/features/voice/useVoicePreferences";
import { useVoiceSession } from "@/features/voice/useVoiceSession";

const STATUS_LABELS: Record<RuntimeStatus, string> = {
  idle: "Talk to Izwi",
  preparing: "Getting ready",
  listening: "Listening",
  user_speaking: "Listening",
  processing: "Thinking",
  assistant_speaking: "Speaking",
};

/** Space and M are left alone while typing or on a focused control. */
function isEditableShortcutTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tagName = target.tagName.toLowerCase();
  return (
    tagName === "input" ||
    tagName === "textarea" ||
    tagName === "select" ||
    tagName === "button" ||
    target.isContentEditable
  );
}

function formatElapsed(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

function useElapsed(startedAt: number | null): string | null {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (startedAt === null) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [startedAt]);
  return startedAt === null ? null : formatElapsed(now - startedAt);
}

export function VoicePage({
  models,
  loading,
  downloadProgress,
  onDownload,
  onCancelDownload,
  onLoad,
  onUnload,
  onError,
}: VoicePageProps) {
  const reportError = useCallback(
    (message: string) => onError?.(message),
    [onError],
  );
  const preferences = useVoicePreferences(reportError);
  const compactModel = resolveCompactVoiceModel(
    preferences.compactModel,
    models,
  );
  const engineRows = useMemo(
    () => voiceEngineModels(preferences.engine, models, compactModel),
    [compactModel, models, preferences.engine],
  );
  const isReady = isVoiceEngineReady(engineRows);
  const speakers = useMemo(
    () => voiceEngineSpeakers(preferences.engine, compactModel),
    [compactModel, preferences.engine],
  );
  const speaker = resolveVoiceSpeaker(speakers, preferences.speaker);
  const sessionEngine = useMemo(
    () =>
      isReady ? voiceSessionEngine(preferences.engine, compactModel) : null,
    [compactModel, isReady, preferences.engine],
  );
  const session = useVoiceSession({
    engine: sessionEngine,
    // None when the engine picks its own voice (LFM3 Audio).
    speaker: speaker?.id,
    systemPrompt: preferences.savedPrompt,
  });

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [settingsView, setSettingsView] = useState<VoiceSettingsView>("main");
  const [isTranscriptOpen, setIsTranscriptOpen] = useState(false);
  const elapsed = useElapsed(session.startedAt);

  const openSettings = useCallback((view: VoiceSettingsView = "main") => {
    setSettingsView(view);
    setIsSettingsOpen(true);
  }, []);

  const { isActive, start, stop, toggleMicMuted } = session;
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        isSettingsOpen ||
        event.defaultPrevented ||
        event.repeat ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        isEditableShortcutTarget(event.target)
      ) {
        return;
      }
      if (event.code === "Space") {
        if (isActive) {
          event.preventDefault();
          stop();
        } else if (isReady) {
          event.preventDefault();
          void start();
        }
        return;
      }
      if (event.key === "Escape" && isActive) {
        event.preventDefault();
        stop();
        return;
      }
      if (event.key.toLowerCase() === "m" && isActive) {
        event.preventDefault();
        toggleMicMuted();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isActive, isReady, isSettingsOpen, start, stop, toggleMicMuted]);

  if (loading) {
    return (
      <PageShell className="flex h-full items-center justify-center">
        <p className="text-sm text-(--text-muted)">Loading models…</p>
      </PageShell>
    );
  }

  const lastLine = session.transcript[session.transcript.length - 1];
  const showsLastLine = !session.notice && !!lastLine?.text;
  const caption = session.notice
    ? session.notice
    : session.isAwaitingMicrophone
      ? "Allow microphone access if asked."
      : session.status === "preparing"
        ? "Loading the voice models…"
        : lastLine?.text
          ? lastLine.text
          : session.status === "listening"
            ? "Go ahead, or press Space to end."
            : session.status === "assistant_speaking"
              ? "Start talking to interrupt"
              : "";

  let stage: ReactNode;
  if (isActive) {
    stage = (
      <div className="flex flex-col items-center gap-9 text-center">
        <VoiceOrb status={session.status} audioLevel={session.audioLevel} />
        <div className="flex min-h-24 max-w-[560px] flex-col items-center gap-2.5">
          <p
            role="status"
            className="text-xl font-medium tracking-[-0.01em] text-(--text-primary)"
          >
            {session.isMicMuted ? "Muted" : STATUS_LABELS[session.status]}
          </p>
          {isTranscriptOpen ? (
            session.status === "assistant_speaking" && (
              <p className="text-sm text-(--text-subtle)">
                Start talking to interrupt
              </p>
            )
          ) : caption ? (
            <p
              className={
                session.notice
                  ? "text-sm text-(--danger-text)"
                  : showsLastLine
                    ? "line-clamp-2 text-lg leading-relaxed text-(--text-secondary)"
                    : "text-sm text-(--text-subtle)"
              }
            >
              {caption}
            </p>
          ) : null}
        </div>
      </div>
    );
  } else if (session.failure) {
    stage = (
      <VoiceFailure
        failure={session.failure}
        onRetry={() => void start()}
        onOpenSettings={() => openSettings("engine")}
      />
    );
  } else if (!isReady) {
    stage = (
      <VoiceSetup
        rows={engineRows}
        downloadProgress={downloadProgress}
        downloadBytes={voiceEngineDownloadBytes(engineRows)}
        onDownload={() =>
          engineRows.forEach((row) => {
            if (
              row.model &&
              (row.model.status === "not_downloaded" ||
                row.model.status === "error")
            ) {
              onDownload(row.variant);
            }
          })
        }
        onCancel={
          onCancelDownload
            ? () =>
                engineRows.forEach((row) => {
                  if (row.model?.status === "downloading") {
                    onCancelDownload(row.variant);
                  }
                })
            : undefined
        }
        onUseCompact={
          preferences.engine === "standard"
            ? () => preferences.setEngine("compact")
            : undefined
        }
        compactLabel={formatModelVariantLabel(compactModel)}
      />
    );
  } else {
    stage = (
      <div className="flex flex-col items-center gap-7 text-center">
        <button
          type="button"
          onClick={() => void start()}
          tabIndex={-1}
          aria-hidden="true"
          className="rounded-full"
        >
          <VoiceOrb status="idle" audioLevel={0} />
        </button>
        <div className="flex flex-col items-center gap-2">
          <h1 className="text-2xl font-medium tracking-[-0.02em] text-(--text-primary)">
            Talk to Izwi
          </h1>
          {speaker && (
            <p className="text-sm text-(--text-muted)">
              Replies in the voice of{" "}
              <button
                type="button"
                onClick={() => openSettings("main")}
                className="text-(--text-primary) underline decoration-(--border-strong) underline-offset-[3px] hover:text-(--text-muted)"
              >
                {speaker.name}
              </button>
            </p>
          )}
        </div>
        <div className="flex flex-col items-center gap-3">
          <button
            type="button"
            onClick={() => void start()}
            className="flex h-12 items-center gap-2 rounded-full bg-(--accent-solid) px-6 text-[15px] font-medium text-(--text-on-accent) transition-opacity hover:opacity-90 focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-(--border-strong) focus-visible:ring-offset-2 focus-visible:ring-offset-(--bg-surface-0)"
          >
            <Mic className="h-[18px] w-[18px]" />
            Start talking
          </button>
          <p className="flex items-center gap-1.5 text-xs text-(--text-subtle)">
            or press
            <kbd className="rounded-md border border-(--border-default) bg-(--bg-surface-2) px-1.5 py-0.5 font-sans text-[11px] text-(--text-muted)">
              Space
            </kbd>
          </p>
        </div>
      </div>
    );
  }

  return (
    <PageShell className="flex h-full min-h-[560px] flex-col lg:flex-row">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <header className="flex h-16 shrink-0 items-center justify-between px-5">
          <span className="text-[13px] tabular-nums text-(--text-subtle)">
            {isActive ? elapsed : ""}
          </span>
          <button
            type="button"
            onClick={() => openSettings("main")}
            aria-label="Voice settings"
            title="Voice settings"
            className="flex h-10 w-10 items-center justify-center rounded-[10px] text-(--text-muted) transition-colors hover:bg-(--bg-surface-2) hover:text-(--text-primary)"
          >
            <SlidersHorizontal className="h-[18px] w-[18px]" />
          </button>
        </header>
        <section className="flex min-h-0 flex-1 flex-col items-center justify-center px-6 pb-10">
          {stage}
        </section>
        {isActive && (
          <footer className="px-6 pb-10 pt-2">
            <VoiceControls
              isMicMuted={session.isMicMuted}
              isTranscriptOpen={isTranscriptOpen}
              onToggleMic={toggleMicMuted}
              onEnd={stop}
              onToggleTranscript={() => setIsTranscriptOpen((open) => !open)}
            />
          </footer>
        )}
      </div>
      {isActive && isTranscriptOpen && (
        <VoiceTranscript
          entries={session.transcript}
          isAssistantLive={
            session.status === "processing" ||
            session.status === "assistant_speaking"
          }
          onClose={() => setIsTranscriptOpen(false)}
        />
      )}
      <VoiceSettingsSheet
        open={isSettingsOpen}
        onOpenChange={setIsSettingsOpen}
        view={settingsView}
        onViewChange={setSettingsView}
        preferences={preferences}
        compactModel={compactModel}
        speakers={speakers}
        speaker={speaker}
        models={models}
        downloadProgress={downloadProgress}
        onDownload={onDownload}
        onLoad={onLoad}
        onUnload={onUnload}
        onError={reportError}
        isConversationActive={isActive}
      />
    </PageShell>
  );
}
