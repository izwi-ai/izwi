import { useCallback, useEffect, useState } from "react";

import { api, type ModelInfo, type SavedVoiceSummary, type StudioProjectSummary } from "@/api";
import { ConfirmDeleteDialog } from "@/components/history/ConfirmDeleteDialog";
import { NewItemButton } from "@/components/history/NewItemButton";
import { PageHeader } from "@/components/PageShell";
import { useNewItemShortcut } from "@/hooks/useNewItemShortcut";
import { NewStudioProjectDialog } from "@/features/studio/components/NewStudioProjectDialog";
import { StudioProjectList } from "@/features/studio/components/StudioProjectList";
import type { StudioModelDownloads } from "@/features/studio/components/StudioVoicePicker";

const PAGE_LIMIT = 24;
/** How often the list refreshes while a project renders. */
const RENDERING_REFRESH_MS = 3000;

interface StudioProjectsPageProps {
  models: ModelInfo[];
  savedVoices: SavedVoiceSummary[];
  downloads: StudioModelDownloads;
  onError: (message: string) => void;
  onOpenProject: (projectId: string) => void;
}

export function StudioProjectsPage({
  models,
  savedVoices,
  downloads,
  onError,
  onOpenProject,
}: StudioProjectsPageProps) {
  const [projects, setProjects] = useState<StudioProjectSummary[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<StudioProjectSummary | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const loadFirstPage = useCallback(async () => {
    try {
      const page = await api.listStudioProjectPage({ limit: PAGE_LIMIT });
      setProjects(page.items);
      setNextCursor(page.pagination.has_more ? page.pagination.next_cursor : null);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't load your projects.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadFirstPage();
  }, [loadFirstPage]);

  const rendering = projects.some((project) => project.pending_segment_count > 0);
  useEffect(() => {
    if (!rendering) return;
    const timer = window.setInterval(() => void loadFirstPage(), RENDERING_REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [loadFirstPage, rendering]);

  const loadMore = async () => {
    if (!nextCursor) return;
    setLoadingMore(true);
    try {
      const page = await api.listStudioProjectPage({ limit: PAGE_LIMIT, cursor: nextCursor });
      setProjects((current) => [
        ...current,
        ...page.items.filter((item) => !current.some((existing) => existing.id === item.id)),
      ]);
      setNextCursor(page.pagination.has_more ? page.pagination.next_cursor : null);
    } catch (err) {
      onError(err instanceof Error ? err.message : "Couldn't load more projects.");
    } finally {
      setLoadingMore(false);
    }
  };

  const confirmDelete = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await api.deleteStudioProject(deleteTarget.id);
      setProjects((current) => current.filter((project) => project.id !== deleteTarget.id));
      setDeleteTarget(null);
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : "Couldn't delete the project.");
    } finally {
      setDeleting(false);
    }
  };

  const openNewProject = useCallback(() => setCreating(true), []);
  useNewItemShortcut(openNewProject);
  const newButton = <NewItemButton label="New project" onClick={openNewProject} />;

  return (
    <>
      <PageHeader
        title="Studio"
        description="Long scripts, narrated paragraph by paragraph."
        actions={projects.length > 0 ? newButton : null}
      />
      <StudioProjectList
        projects={projects}
        savedVoices={savedVoices}
        loading={loading}
        error={error}
        onRetry={() => void loadFirstPage()}
        loadMore={{
          canLoadMore: Boolean(nextCursor),
          loading: loadingMore,
          onLoadMore: () => void loadMore(),
        }}
        emptyAction={newButton}
        onOpen={onOpenProject}
        onDelete={(project) => {
          setDeleteError(null);
          setDeleteTarget(project);
        }}
      />
      <NewStudioProjectDialog
        open={creating}
        onClose={() => setCreating(false)}
        models={models}
        savedVoices={savedVoices}
        downloads={downloads}
        onError={onError}
        onCreated={(project) => {
          setCreating(false);
          onOpenProject(project.id);
        }}
      />
      <ConfirmDeleteDialog
        noun={deleteTarget ? "project" : null}
        itemLabel={deleteTarget?.name}
        description="Its script and audio are deleted. This can't be undone."
        pending={deleting}
        error={deleteError}
        onConfirm={() => void confirmDelete()}
        onCancel={() => setDeleteTarget(null)}
      />
    </>
  );
}
