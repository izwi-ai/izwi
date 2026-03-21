import { useEffect, useState } from "react";
import { ArrowLeft, AudioLines, Sparkles } from "lucide-react";
import { VoiceCaptureWorkspace } from "@/components/VoiceCaptureWorkspace";
import { VoiceDesignWorkspace } from "@/components/VoiceDesignWorkspace";
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
  onUseSavedVoiceInTts?: (voiceId: string) => void;
  designModel: string | null;
  designModelReady: boolean;
  designModelOptions: Array<{
    value: string;
    label: string;
    statusLabel: string;
    isReady: boolean;
  }>;
  onSelectDesignModel?: (variant: string) => void;
  onOpenDesignModelManager?: () => void;
  onDesignModelRequired: () => void;
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

export function VoiceCreationModal({
  open,
  onOpenChange,
  onUseSavedVoiceInTts,
  designModel,
  designModelReady,
  designModelOptions,
  onSelectDesignModel,
  onOpenDesignModelManager,
  onDesignModelRequired,
}: VoiceCreationModalProps) {
  const [step, setStep] = useState<CreationStep>("choice");
  const [hasDraftProgress, setHasDraftProgress] = useState(false);
  const [savedVoiceResult, setSavedVoiceResult] = useState<{
    voiceId: string;
    source: "clone" | "design";
  } | null>(null);

  useEffect(() => {
    if (!open) {
      setStep("choice");
      setHasDraftProgress(false);
      setSavedVoiceResult(null);
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
    setSavedVoiceResult(null);
  };

  const handleSavedVoice = (voiceId: string, source: "clone" | "design") => {
    setSavedVoiceResult({ voiceId, source });
    setHasDraftProgress(false);
    setStep("success");
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
              {step !== "choice" && step !== "success" ? (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 rounded-[var(--radius-pill)] px-3 text-xs"
                  onClick={() => {
                    setStep("choice");
                    setHasDraftProgress(false);
                    setSavedVoiceResult(null);
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
                onVoiceSaved={(voiceId) => handleSavedVoice(voiceId, "clone")}
              />
            </div>
          ) : null}

          {step === "design" ? (
            <div className="space-y-3">
              <VoiceDesignWorkspace
                selectedModel={designModel}
                selectedModelReady={designModelReady}
                modelOptions={designModelOptions}
                onSelectModel={onSelectDesignModel}
                onOpenModelManager={onOpenDesignModelManager}
                onModelRequired={onDesignModelRequired}
                embeddedInModal
                onVoiceSaved={(voiceId) => handleSavedVoice(voiceId, "design")}
              />
            </div>
          ) : null}

          {step === "success" ? (
            <div className="space-y-4">
              <div className="rounded-xl border border-[var(--status-positive-border)] bg-[var(--status-positive-bg)] px-4 py-4">
                <h3 className="text-sm font-semibold text-[var(--status-positive-text)]">
                  Voice saved successfully
                </h3>
                <p className="mt-1 text-sm text-[var(--status-positive-text)]/90">
                  {savedVoiceResult?.source === "design"
                    ? "Your designed voice is ready in the library and available for Text to Speech."
                    : "Your cloned voice is ready in the library and available for Text to Speech."}
                </p>
              </div>

              <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                <Button
                  className="h-10 rounded-[var(--radius-pill)] px-4 text-sm"
                  disabled={!savedVoiceResult || !onUseSavedVoiceInTts}
                  onClick={() => {
                    if (!savedVoiceResult || !onUseSavedVoiceInTts) {
                      return;
                    }
                    onUseSavedVoiceInTts(savedVoiceResult.voiceId);
                    onOpenChange(false);
                  }}
                >
                  Use in Text to Speech
                </Button>
                <Button
                  variant="outline"
                  className="h-10 rounded-[var(--radius-pill)] border-[var(--border-strong)] px-4 text-sm"
                  onClick={() => {
                    setStep("choice");
                    setSavedVoiceResult(null);
                  }}
                >
                  Create Another
                </Button>
                <Button
                  variant="ghost"
                  className="h-10 rounded-[var(--radius-pill)] px-4 text-sm"
                  onClick={() => onOpenChange(false)}
                >
                  Back to Library
                </Button>
              </div>
            </div>
          ) : null}
        </div>
      </DialogContent>
    </Dialog>
  );
}
