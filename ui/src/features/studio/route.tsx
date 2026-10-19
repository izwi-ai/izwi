import { useMemo } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { AlertTriangle, Loader2 } from "lucide-react";

import type { SharedPageProps } from "@/app/router/types";
import { usePageTitle } from "@/app/router/DocumentTitle";
import { PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { StudioEditor } from "@/features/studio/components/StudioEditor";
import { StudioProjectsPage } from "@/features/studio/components/StudioProjectsPage";
import type { StudioModelDownloads } from "@/features/studio/components/StudioVoicePicker";
import { useSavedVoices } from "@/features/studio/hooks/useSavedVoices";
import { useStudioProject } from "@/features/studio/hooks/useStudioProject";
import { studioModels } from "@/features/studio/studio";

/** `/studio` lists projects; `/studio/:projectId` edits one. */
export function StudioPage({ models, downloadProgress, onDownload, onError }: SharedPageProps) {
  const navigate = useNavigate();
  const { projectId } = useParams<{ projectId?: string }>();
  const savedVoices = useSavedVoices();
  const renderable = useMemo(() => studioModels(models), [models]);
  const downloads: StudioModelDownloads = { downloadProgress, onDownload };

  return (
    <PageShell className="pb-0">
      {projectId ? (
        <StudioProjectRoute
          key={projectId}
          projectId={projectId}
          models={renderable}
          savedVoices={savedVoices}
          downloads={downloads}
          onError={onError}
          onDeleted={() => navigate("/studio")}
        />
      ) : (
        <StudioProjectsPage
          models={renderable}
          savedVoices={savedVoices}
          downloads={downloads}
          onError={onError}
          onOpenProject={(id) => navigate(`/studio/${id}`)}
        />
      )}
    </PageShell>
  );
}

function StudioProjectRoute({
  projectId,
  models,
  savedVoices,
  downloads,
  onError,
  onDeleted,
}: {
  projectId: string;
  models: SharedPageProps["models"];
  savedVoices: ReturnType<typeof useSavedVoices>;
  downloads: StudioModelDownloads;
  onError: (message: string) => void;
  onDeleted: () => void;
}) {
  const state = useStudioProject(projectId, onError);
  usePageTitle(state.project?.name ?? null);

  if (state.project) {
    return (
      <StudioEditor
        state={{ ...state, project: state.project }}
        models={models}
        savedVoices={savedVoices}
        downloads={downloads}
        onError={onError}
        onDeleted={onDeleted}
      />
    );
  }
  if (state.load === "loading") {
    return (
      <div className="flex min-h-[50vh] items-center justify-center text-sm text-[var(--text-muted)]">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Opening project…
      </div>
    );
  }
  return (
    <div className="mx-auto flex min-h-[50vh] max-w-md flex-col items-center justify-center gap-4 text-center">
      <AlertTriangle className="h-5 w-5 text-[var(--text-muted)]" />
      <div className="flex flex-col gap-1.5">
        <h1 className="text-base font-semibold text-[var(--text-primary)]">
          {state.load === "missing" ? "This project doesn't exist" : "Couldn't open this project"}
        </h1>
        <p className="text-sm text-[var(--text-muted)]">
          {state.load === "missing"
            ? "It may have been deleted."
            : (state.loadError ?? "Try again in a moment.")}
        </p>
      </div>
      <div className="flex gap-2">
        {state.load === "error" ? (
          <Button type="button" variant="outline" size="sm" onClick={state.reload}>
            Try again
          </Button>
        ) : null}
        <Button asChild size="sm">
          <Link to="/studio">Back to Studio</Link>
        </Button>
      </div>
    </div>
  );
}
