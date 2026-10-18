import { CircleAlert } from "lucide-react";

import type { VoiceSessionFailure } from "@/features/voice/useVoiceSession";

const TITLES: Record<VoiceSessionFailure["kind"], string> = {
  connection: "Conversation ended",
  microphone: "Microphone is blocked",
  model: "Couldn't load the voice models",
  other: "Conversation ended",
};

/** Why the last conversation stopped, and the way back in. */
export function VoiceFailure({
  failure,
  onRetry,
  onOpenSettings,
}: {
  failure: VoiceSessionFailure;
  onRetry: () => void;
  onOpenSettings: () => void;
}) {
  return (
    <div className="flex flex-col items-center gap-7 text-center">
      <div
        aria-hidden="true"
        className="flex h-[184px] w-[184px] items-center justify-center rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] text-[var(--danger-text)]"
      >
        <CircleAlert className="h-8 w-8" strokeWidth={1.75} />
      </div>
      <div role="alert" className="flex max-w-[440px] flex-col items-center gap-2">
        <h1 className="text-2xl font-medium tracking-[-0.02em] text-[var(--text-primary)]">
          {TITLES[failure.kind]}
        </h1>
        <p className="text-sm leading-relaxed text-[var(--text-muted)]">
          {failure.message}
        </p>
      </div>
      <div className="flex flex-wrap justify-center gap-2.5">
        <button
          type="button"
          onClick={onRetry}
          className="h-11 rounded-full bg-[var(--accent-solid)] px-[22px] text-sm font-medium text-[var(--text-on-accent)] transition-opacity hover:opacity-90"
        >
          {failure.kind === "microphone" ? "Try again" : "Start again"}
        </button>
        {failure.kind === "model" && (
          <button
            type="button"
            onClick={onOpenSettings}
            className="h-11 rounded-full border border-[var(--border-default)] px-[18px] text-sm text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-surface-2)]"
          >
            Change engine
          </button>
        )}
      </div>
    </div>
  );
}
