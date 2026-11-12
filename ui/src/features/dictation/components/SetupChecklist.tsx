import { useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";

import type { ModelInfo } from "@/api";
import { SetupRow, downloadPercent, type SetupRowTone } from "@/components/SetupRow";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { dictationApi } from "@/features/dictation/api";
import type {
  DictationSnapshot,
  DictationStatus,
} from "@/features/dictation/types";
import { getModelDisplayName } from "@/features/models/catalog/modelMetadata";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";

type ModelStep =
  | "ready"
  | "downloading"
  | "loading"
  | "not_downloaded"
  | "error"
  | "checking";

/**
 * Where the speech model stands. The snapshot says whether dictation can use
 * it; the catalog knows about a download this page started before the
 * snapshot catches up.
 */
export function speechModelStep(
  snapshot: DictationSnapshot,
  catalogModel: ModelInfo | undefined,
): ModelStep {
  const { state } = snapshot.model;
  if (state === "ready") {
    return "ready";
  }
  if (state === "downloading" || catalogModel?.status === "downloading") {
    return "downloading";
  }
  if (state === "loading" || state === "error") {
    return state;
  }
  if (state === "not_downloaded") {
    // On disk already: the app is about to notice.
    return catalogModel && catalogModel.status !== "not_downloaded"
      ? catalogModel.status === "error"
        ? "not_downloaded"
        : "checking"
      : "not_downloaded";
  }
  return "checking";
}

/** The speech model no longer holds dictation up. */
export function speechModelSettled(snapshot: DictationSnapshot): boolean {
  return (
    snapshot.model.state === "ready" ||
    // A remote server's models cannot be checked from here.
    (snapshot.server_is_remote && snapshot.model.state === "unknown")
  );
}

export function keyboardPermissionSettled(snapshot: DictationSnapshot): boolean {
  return (
    !snapshot.platform.needs_keyboard_permission || snapshot.permissions.keyboard
  );
}

const MODEL_STATUS: Record<ModelStep, { label: string; tone: SetupRowTone }> = {
  ready: { label: "Ready", tone: "done" },
  downloading: { label: "Downloading", tone: "active" },
  loading: { label: "Loading", tone: "active" },
  not_downloaded: { label: "Not downloaded", tone: "muted" },
  error: { label: "Needs attention", tone: "error" },
  checking: { label: "Checking", tone: "muted" },
};

function tryButtonLabel(status: DictationStatus): string {
  switch (status.state) {
    case "listening":
      return "Stop";
    case "starting":
      return "Starting…";
    case "transcribing":
      return "Transcribing…";
    default:
      return "Start";
  }
}

interface SetupChecklistProps {
  snapshot: DictationSnapshot;
  partial: string;
  models: ModelInfo[];
  downloadProgress: ModelDownloadProgressMap;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onRefresh: () => Promise<void>;
}

/** The steps before dictation works: model, permission, a first try. */
export function SetupChecklist({
  snapshot,
  partial,
  models,
  downloadProgress,
  onDownload,
  onCancelDownload,
  onRefresh,
}: SetupChecklistProps) {
  const modelId = snapshot.model.id;
  const catalogModel = models.find((model) => model.variant === modelId);
  const step = speechModelStep(snapshot, catalogModel);
  const modelStatus = MODEL_STATUS[step];
  const [tryText, setTryText] = useState("");
  const [actionError, setActionError] = useState<string | null>(null);
  const tryAreaRef = useRef<HTMLTextAreaElement>(null);

  // A download that finished here: the snapshot read makes the app look at
  // the model again.
  const catalogStatus = catalogModel?.status;
  const lastCatalogStatusRef = useRef(catalogStatus);
  useEffect(() => {
    if (lastCatalogStatusRef.current === catalogStatus) {
      return;
    }
    lastCatalogStatusRef.current = catalogStatus;
    if (catalogStatus && catalogStatus !== "downloading") {
      void onRefresh();
    }
  }, [catalogStatus, onRefresh]);

  const run = async (action: () => Promise<unknown>) => {
    setActionError(null);
    try {
      await action();
    } catch (error) {
      setActionError(typeof error === "string" ? error : "That didn't work.");
    }
  };

  const status = snapshot.status;
  const busy = status.state === "starting" || status.state === "transcribing";
  const canTry =
    snapshot.settings.enabled && speechModelSettled(snapshot) && !busy;

  return (
    <ul className="[&>li:last-child]:border-b-0">
      <SetupRow
        title="Speech model"
        model={getModelDisplayName(modelId, catalogModel?.display_name)}
        status={
          step === "downloading" && catalogModel
            ? `Downloading ${downloadPercent(catalogModel, downloadProgress)}%`
            : modelStatus.label
        }
        tone={modelStatus.tone}
        percent={
          step === "downloading" && catalogModel
            ? downloadPercent(catalogModel, downloadProgress)
            : undefined
        }
        action={
          step === "not_downloaded" ? (
            <Button type="button" size="sm" onClick={() => onDownload(modelId)}>
              Download
            </Button>
          ) : step === "downloading" && onCancelDownload ? (
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => onCancelDownload(modelId)}
            >
              Cancel
            </Button>
          ) : step === "loading" || step === "checking" ? (
            <Loader2
              aria-hidden="true"
              className="h-4 w-4 animate-spin text-(--text-subtle)"
            />
          ) : step === "error" ? (
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => void onRefresh()}
            >
              Check again
            </Button>
          ) : undefined
        }
        detail={
          step === "error"
            ? (snapshot.model.message ?? "The speech model could not be loaded.")
            : step === "not_downloaded"
              ? "Dictation transcribes on this computer with this model."
              : undefined
        }
      />

      {snapshot.platform.needs_keyboard_permission ? (
        <SetupRow
          title="Keyboard control"
          model="Accessibility permission"
          status={snapshot.permissions.keyboard ? "Allowed" : "Not allowed"}
          tone={snapshot.permissions.keyboard ? "done" : "muted"}
          action={
            snapshot.permissions.keyboard ? undefined : (
              <>
                <Button
                  type="button"
                  size="sm"
                  onClick={() =>
                    void run(async () => {
                      await dictationApi.requestKeyboardPermission();
                      await onRefresh();
                    })
                  }
                >
                  Allow…
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() =>
                    void run(() => dictationApi.openSystemSettings("accessibility"))
                  }
                >
                  Open System Settings
                </Button>
              </>
            )
          }
          detail={
            snapshot.permissions.keyboard
              ? undefined
              : "macOS needs Accessibility permission so Izwi can paste for you. Without it, your words are copied and you paste them yourself."
          }
        />
      ) : null}

      <li className="flex flex-col gap-2.5 border-b border-(--border-muted) py-3.5">
        <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-4">
          <div className="flex min-w-0 flex-col gap-0.5">
            <span className="text-sm font-medium text-(--text-primary)">
              Try it
            </span>
            <span className="text-[13px] text-(--text-muted)">
              Click in the box, then use your shortcut or the button.
            </span>
          </div>
          <span className="flex items-center gap-2">
            {status.state === "listening" ? (
              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={() => void run(() => dictationApi.cancel())}
              >
                Cancel
              </Button>
            ) : null}
            <Button
              type="button"
              size="sm"
              variant={status.state === "listening" ? "default" : "outline"}
              disabled={!canTry}
              onClick={() => {
                // The words go wherever the cursor is: put it in the box.
                tryAreaRef.current?.focus();
                void run(() => dictationApi.toggle());
              }}
            >
              {tryButtonLabel(status)}
            </Button>
          </span>
        </div>
        <Textarea
          ref={tryAreaRef}
          value={tryText}
          onChange={(event) => setTryText(event.target.value)}
          aria-label="Try dictation"
          placeholder="Your words appear here."
          rows={3}
          className="rounded-[10px] border-(--border-default) bg-(--surface-panel) text-[13.5px] shadow-none"
        />
        {partial && status.state === "listening" ? (
          <p className="truncate text-[12.5px] text-(--text-subtle)">{partial}</p>
        ) : null}
        {actionError ? (
          <p role="alert" className="text-[12.5px] text-(--danger-text)">
            {actionError}
          </p>
        ) : null}
      </li>
    </ul>
  );
}
