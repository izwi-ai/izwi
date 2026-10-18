import { ExternalLink, FileAudio, Trash2 } from "lucide-react";

import type { StudioProjectMetaRecord, StudioProjectSummary } from "@/api";
import { formatHistoryDate } from "@/components/history/format";
import { HistoryTable, type HistoryColumn } from "@/components/history/HistoryTable";
import { RowBadge } from "@/components/history/RowBadge";
import { RowMenu } from "@/components/history/RowMenu";
import { Button } from "@/components/ui/button";
import { getModelDisplayName } from "@/features/models/catalog/modelMetadata";

interface StudioProjectHistoryTableProps {
  projects: StudioProjectSummary[];
  projectMetaById?: Record<string, StudioProjectMetaRecord>;
  loading?: boolean;
  error?: string | null;
  deletePending?: boolean;
  loadMore?: {
    canLoadMore: boolean;
    loading: boolean;
    onLoadMore: () => void;
  };
  onRefresh?: () => void;
  onCreateProject?: () => void;
  onOpenProject: (projectId: string) => void;
  onRequestDeleteProject?: (project: StudioProjectSummary) => void;
}

function rowLabel(project: StudioProjectSummary): string {
  return project.name || project.model_id || project.id;
}

function ProgressCell({ project }: { project: StudioProjectSummary }) {
  const fraction =
    project.segment_count > 0
      ? Math.min(project.rendered_segment_count / project.segment_count, 1)
      : 0;
  const done = fraction >= 1;
  return (
    <div className="flex items-center gap-2.5">
      <div
        role="progressbar"
        aria-label="Rendered segments"
        aria-valuemin={0}
        aria-valuemax={project.segment_count}
        aria-valuenow={project.rendered_segment_count}
        className="h-1 w-16 overflow-hidden rounded-full bg-[var(--surface-raised)]"
      >
        <div
          className={
            done
              ? "h-full bg-[var(--status-positive-solid)]"
              : "h-full bg-[var(--text-muted)]"
          }
          style={{ width: `${fraction * 100}%` }}
        />
      </div>
      <span className="whitespace-nowrap text-[13px] tabular-nums text-[var(--text-muted)]">
        {project.rendered_segment_count}/{project.segment_count}
      </span>
      {done ? (
        <RowBadge tone="positive">Ready</RowBadge>
      ) : project.rendered_segment_count === 0 ? (
        <RowBadge>Not rendered</RowBadge>
      ) : null}
    </div>
  );
}

export function StudioProjectHistoryTable({
  projects,
  projectMetaById = {},
  loading = false,
  error = null,
  deletePending = false,
  loadMore,
  onRefresh,
  onCreateProject,
  onOpenProject,
  onRequestDeleteProject,
}: StudioProjectHistoryTableProps) {
  const columns: HistoryColumn<StudioProjectSummary>[] = [
    {
      id: "project",
      header: "Project",
      className: "max-w-0",
      cell: (project) => {
        const tags = projectMetaById[project.id]?.tags ?? [];
        return (
          <div className="flex min-w-0 items-center gap-2.5">
            <span className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-[9px] border border-[var(--border-subtle)] bg-[var(--surface-panel-alt)] text-[var(--text-muted)]">
              <FileAudio className="h-4 w-4" />
            </span>
            <div className="min-w-0">
              <div className="truncate font-medium text-[var(--text-primary)]">
                {project.name}
              </div>
              <div className="truncate text-xs text-[var(--text-subtle)]">
                {tags.length > 0
                  ? tags.slice(0, 3).join(" · ")
                  : project.source_filename || "Script project"}
              </div>
            </div>
          </div>
        );
      },
    },
    {
      id: "progress",
      header: "Segments",
      className: "w-[15rem]",
      cell: (project) => <ProgressCell project={project} />,
    },
    {
      id: "chars",
      header: "Characters",
      className: "hidden w-[7rem] lg:table-cell",
      cellClassName: "whitespace-nowrap tabular-nums text-[var(--text-muted)]",
      cell: (project) => project.total_chars.toLocaleString(),
    },
    {
      id: "model",
      header: "Model",
      className: "w-[11rem]",
      cellClassName: "truncate text-[13px] text-[var(--text-secondary)]",
      cell: (project) =>
        project.model_id ? getModelDisplayName(project.model_id) : "Not set",
    },
    {
      id: "updated",
      header: "Edited",
      className: "hidden w-[8rem] lg:table-cell",
      cellClassName: "whitespace-nowrap text-[13px] text-[var(--text-subtle)]",
      cell: (project) =>
        formatHistoryDate(project.updated_at || project.created_at),
    },
  ];

  return (
    <HistoryTable
      rows={projects}
      columns={columns}
      getRowId={(project) => project.id}
      getRowLabel={(project) => `Studio project ${rowLabel(project)}`}
      onOpenRow={(project) => onOpenProject(project.id)}
      loading={loading}
      error={error}
      onRetry={onRefresh}
      total={null}
      noun="projects"
      empty={{
        title: "No Studio projects yet",
        description:
          "Studio turns long scripts into narrated audio, segment by segment.",
        action: onCreateProject ? (
          <Button type="button" onClick={onCreateProject}>
            New project
          </Button>
        ) : undefined,
      }}
      loadMore={loadMore}
      rowActions={(project) => (
        <RowMenu
          label={rowLabel(project)}
          items={[
            {
              id: "open",
              label: "Open project",
              icon: ExternalLink,
              onSelect: () => onOpenProject(project.id),
            },
            {
              id: "delete",
              label: "Delete",
              icon: Trash2,
              danger: true,
              separatorBefore: true,
              disabled: !onRequestDeleteProject || deletePending,
              onSelect: () => onRequestDeleteProject?.(project),
            },
          ]}
        />
      )}
    />
  );
}
