import { useEffect, useState } from "react";
import { ArrowLeft, AudioLines, Sparkles } from "lucide-react";
import { VoiceCaptureWorkspace } from "@/components/VoiceCaptureWorkspace";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

type CreationStep = "choice" | "clone" | "design" | "success";

interface VoiceCreationModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function stepTitle(step: CreationStep): string {
  if (step === "clone") {
    return "Clone Voice";
  }
  if (step === "design") {
    return "Design Voice";
  }
  if (step === "success") {
    return "Voice Saved";
  }
  return "New Voice";
}

function stepDescription(step: CreationStep): string {
  if (step === "clone") {
    return "Create a reusable voice from uploaded or recorded reference speech.";
  }
  if (step === "design") {
    return "Generate a brand new voice profile from a natural-language prompt.";
  }
  if (step === "success") {
    return "Your voice is ready to use in Text to Speech.";
  }
  return "Choose how you want to create a new reusable voice profile.";
}

export function VoiceCreationModal({ open, onOpenChange }: VoiceCreationModalProps) {
  const [step, setStep] = useState<CreationStep>("choice");
  const [hasDraftProgress, setHasDraftProgress] = useState(false);
  const [savedCloneVoiceId, setSavedCloneVoiceId] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      setStep("choice");
      setHasDraftProgress(false);
      setSavedCloneVoiceId(null);
    }
  }, [open]);

  const requestClose = () => {
    if (hasDraftProgress && step !== "choice" && step !== "success") {
      const shouldDiscard = window.confirm(
        "Discard your current voice creation progress?",
      );
      if (!shouldDiscard) {
        return;
      }
    }
    onOpenChange(false);
  };

  const handleStepSelect = (nextStep: "clone" | "design") => {
    setStep(nextStep);
    setHasDraftProgress(true);
    setSavedCloneVoiceId(null);
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen) {
          requestClose();
          return;
        }
        onOpenChange(nextOpen);
      }}
    >
      <DialogContent className="max-h-[88vh] overflow-hidden p-0 sm:max-w-[860px]">
        <div className="border-b border-[var(--border-muted)] px-6 py-5">
          <DialogHeader className="gap-2">
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <DialogTitle>{stepTitle(step)}</DialogTitle>
                <DialogDescription className="mt-1">
                  {stepDescription(step)}
                </DialogDescription>
              </div>
              {step !== "choice" ? (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 rounded-[var(--radius-pill)] px-3 text-xs"
                  onClick={() => {
                    setStep("choice");
                    setHasDraftProgress(false);
                  }}
                >
                  <ArrowLeft className="h-3.5 w-3.5" />
                  Back
                </Button>
              ) : null}
            </div>
          </DialogHeader>
        </div>

        <div className="max-h-[calc(88vh-104px)] overflow-y-auto px-6 py-5">
          {step === "choice" ? (
            <div className="grid gap-3 sm:grid-cols-2">
              <button
                type="button"
                onClick={() => handleStepSelect("clone")}
                className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 text-left transition-colors hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-2)]"
              >
                <div className="flex items-center gap-2 text-sm font-semibold text-[var(--text-primary)]">
                  <AudioLines className="h-4 w-4" />
                  Clone Voice
                </div>
                <p className="mt-2 text-sm text-[var(--text-secondary)]">
                  Upload or record a reference sample with transcript, then save it
                  as a reusable voice.
                </p>
              </button>

              <button
                type="button"
                onClick={() => handleStepSelect("design")}
                className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 text-left transition-colors hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-2)]"
              >
                <div className="flex items-center gap-2 text-sm font-semibold text-[var(--text-primary)]">
                  <Sparkles className="h-4 w-4" />
                  Design Voice
                </div>
                <p className="mt-2 text-sm text-[var(--text-secondary)]">
                  Describe a target voice, compare generated candidates, and save
                  the best option.
                </p>
              </button>
            </div>
          ) : null}

          {step === "clone" ? (
            <div className="space-y-3">
              <VoiceCaptureWorkspace
                layout="modal"
                onVoiceSaved={setSavedCloneVoiceId}
              />
              {savedCloneVoiceId ? (
                <div className="rounded-lg border border-[var(--status-positive-border)] bg-[var(--status-positive-bg)] px-3 py-2 text-xs font-medium text-[var(--status-positive-text)]">
                  Saved voice profile is ready in your library.
                </div>
              ) : null}
            </div>
          ) : null}

          {step === "design" ? (
            <div className="rounded-xl border border-dashed border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-8 text-center text-sm text-[var(--text-muted)]">
              Voice design workflow is being moved into this modal in the next
              rollout step.
            </div>
          ) : null}
        </div>
      </DialogContent>
    </Dialog>
  );
}
