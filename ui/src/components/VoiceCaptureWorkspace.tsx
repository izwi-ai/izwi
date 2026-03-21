import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowRight, ShieldCheck, Users } from "lucide-react";
import clsx from "clsx";
import { VoiceClone, type VoiceCloneReferenceState } from "./VoiceClone";
import {
  VOICE_ROUTE_META_COPY_CLASS,
  VOICE_ROUTE_PANEL_TITLE_CLASS,
  VOICE_ROUTE_SECTION_LABEL_CLASS,
} from "@/components/voiceRouteTypography";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/ui/status-badge";
import { WorkspacePanel } from "@/components/ui/workspace";

interface VoiceCaptureWorkspaceProps {
  onUseInTts?: (voiceId: string) => void;
  onVoiceSaved?: (voiceId: string) => void;
  layout?: "page" | "modal";
}

export function VoiceCaptureWorkspace({
  onUseInTts,
  onVoiceSaved,
  layout = "page",
}: VoiceCaptureWorkspaceProps) {
  const [referenceState, setReferenceState] =
    useState<VoiceCloneReferenceState | null>(null);
  const [savedVoiceId, setSavedVoiceId] = useState<string | null>(null);
  const [consentConfirmed, setConsentConfirmed] = useState(false);

  const reusableVoiceId =
    savedVoiceId || referenceState?.activeSavedVoiceId || null;
  const sampleReady = referenceState?.sampleReady ?? false;
  const transcriptChars = referenceState?.transcriptChars ?? 0;
  const sampleDurationSecs = referenceState?.sampleDurationSecs ?? null;
  const referenceWarning = referenceState?.warnings?.[0] ?? null;
  const canUseInTts = Boolean(reusableVoiceId && consentConfirmed && onUseInTts);

  useEffect(() => {
    if (savedVoiceId) {
      onVoiceSaved?.(savedVoiceId);
    }
  }, [onVoiceSaved, savedVoiceId]);

  const handleReferenceReady = useCallback(
    (_audioBase64: string, _transcript: string) => {
      // VoiceClone handles save flows internally; capture tab only needs readiness signals.
    },
    [],
  );

  const handleReferenceClear = useCallback(() => {
    setSavedVoiceId(null);
    setReferenceState(null);
    setConsentConfirmed(false);
  }, []);

  const readinessLabel = useMemo(() => {
    if (sampleReady) {
      return "Reference ready";
    }
    return "Awaiting sample";
  }, [sampleReady]);

  return (
    <div
      className={clsx(
        "grid items-start gap-6",
        layout === "page" ? "pb-4 sm:pb-5" : "pb-1",
      )}
    >
      <WorkspacePanel className="p-4 sm:p-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className={VOICE_ROUTE_SECTION_LABEL_CLASS}>Capture Workflow</div>
            <p className="mt-1 text-sm text-[var(--text-secondary)]">
              Upload or record a reference sample with transcript, then save it as
              a reusable voice for Text to Speech.
            </p>
          </div>
          <StatusBadge tone={sampleReady ? "success" : "warning"}>
            {readinessLabel}
          </StatusBadge>
        </div>
      </WorkspacePanel>

      <WorkspacePanel className="p-6">
        <div className="mb-5 flex items-center gap-2">
          <Users className="h-5 w-5 text-[var(--text-muted)]" />
          <span className={VOICE_ROUTE_PANEL_TITLE_CLASS}>Voice Reference</span>
        </div>

        <VoiceClone
          onVoiceCloneReady={handleReferenceReady}
          onClear={handleReferenceClear}
          onReferenceStateChange={setReferenceState}
          onSavedVoiceCreated={setSavedVoiceId}
        />

        <div className="mt-5 grid gap-4 xl:grid-cols-2">
          <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-5">
            <div className={clsx(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-3")}>
              Quality Checks
            </div>
            <div className="space-y-2.5 text-sm">
              <div className="flex items-center justify-between gap-3">
                <span className="text-[var(--text-secondary)]">Reference length</span>
                <span className="font-medium text-[var(--text-primary)]">
                  {sampleDurationSecs ? `${sampleDurationSecs.toFixed(1)}s` : "No sample yet"}
                </span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="text-[var(--text-secondary)]">Transcript coverage</span>
                <span className="font-medium text-[var(--text-primary)]">
                  {transcriptChars} chars
                </span>
              </div>
              {!sampleReady ? (
                <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2 text-xs text-[var(--text-muted)]">
                  Add a clean reference sample and transcript to save a reusable
                  voice profile.
                </div>
              ) : referenceWarning ? (
                <div className="rounded-lg border border-amber-500/25 bg-amber-500/10 px-3 py-2 text-xs text-amber-700">
                  {referenceWarning}
                </div>
              ) : (
                <div className="rounded-lg border border-green-500/25 bg-green-500/10 px-3 py-2 text-xs text-green-600">
                  Clean sample and transcript are ready to save.
                </div>
              )}
            </div>
          </div>

          <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
            <div className={clsx(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-2")}>
              Rights and Reuse
            </div>

            <label className="flex items-start gap-3 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-3 text-sm">
              <input
                type="checkbox"
                checked={consentConfirmed}
                onChange={(event) => setConsentConfirmed(event.target.checked)}
                className="mt-0.5 h-4 w-4 rounded border-[var(--border-muted)]"
              />
              <span className="text-[var(--text-secondary)]">
                I have permission to clone this voice and use the resulting
                profile in speech generation.
              </span>
            </label>

            <div className="mt-3 flex flex-wrap items-center gap-2">
              {canUseInTts ? (
                <Button
                  size="sm"
                  onClick={() => onUseInTts?.(reusableVoiceId!)}
                  className="h-9 rounded-lg px-4 text-xs"
                >
                  <ArrowRight className="h-4 w-4" />
                  Use in TTS
                </Button>
              ) : (
                <div className={VOICE_ROUTE_META_COPY_CLASS}>
                  {!reusableVoiceId
                    ? "Save a voice profile to enable one-click use in Text to Speech."
                    : "Confirm rights to unlock direct handoff to Text to Speech."}
                </div>
              )}
            </div>

            <div className="mt-3 flex items-center gap-2 text-xs text-[var(--text-muted)]">
              <ShieldCheck className="h-3.5 w-3.5" />
              Rendering and audio generation happen in the Text to Speech route.
            </div>
          </div>
        </div>
      </WorkspacePanel>
    </div>
  );
}
