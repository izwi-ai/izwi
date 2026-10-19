import type { ReactNode } from "react";
import { AlertCircle, Check, ExternalLink, Loader2, Trash2 } from "lucide-react";

import type { SavedVoiceSummary, StudioProjectSummary } from "@/api";
import { formatClockDuration, formatHistoryDate } from "@/components/history/format";
import { HistoryTable, type HistoryColumn } from "@/components/history/HistoryTable";
import { RowMenu } from "@/components/history/RowMenu";
import { voiceName } from "@/features/studio/studio";

interface StudioProjectListProps {
  projects: StudioProjectSummary[];
  savedVoices: SavedVoiceSummary[];
  loading: boolean;
  error: string | null;
  onRetry: () => void;
  loadMore?: { canLoadMore: boolean; loading: boolean; onLoadMore: () => void };
  emptyAction: ReactNode;
  onOpen: (projectId: string) => void;
  onDelete: (project: StudioProjectSummary) => void;
}

/** One line on where a project's audio stands. */
export function ProjectStatus({ project }: { project: StudioProjectSummary }) {
  const {
    segment_count: total,
    rendered_segment_count: rendered,
    stale_segment_count: stale,
    pending_segment_count: pending,
    failed_segment_count: failed,
  } = project;
  if (pending > 0) {
    return (
      <span className="flex items-center gap-2.5 text-[var(--text-secondary)]">
        <span
          role="progressbar"
          aria-label="Rendered paragraphs"
          aria-valuemin={0}
          aria-valuemax={total}
          aria-valuenow={rendered}
          className="h-1 w-16 overflow-hidden rounded-full bg-[var(--surface-raised)]"
        >
          <span
            className="block h-full bg-[var(--text-primary)]"
            style={{ width: `${total > 0 ? (rendered / total) * 100 : 0}%` }}
          />
        </span>
        <Loader2 aria-hidden="true" className="h-3.5 w-3.5 animate-spin" />
        Rendering, {pending} left
      </span>
    );
  }
  if (failed > 0) {
    return (
      <span className="flex items-center gap-2 text-[var(--danger-text)]">
        <AlertCircle aria-hidden="true" className="h-3.5 w-3.5" />
        {failed === 1 ? "1 paragraph failed" : `${failed} paragraphs failed`}
      </span>
    );
  }
  if (stale > 0) {
    return (
      <span className="flex items-center gap-2 text-[var(--status-warning-text)]">
        <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-current" />
        {stale === 1 ? "1 change to render" : `${stale} changes to render`}
      </span>
    );
  }
  if (total > 0 && rendered === total) {
    return (
      <span className="flex items-center gap-2 text-[var(--status-positive-text)]">
        <Check aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={2.5} />
        Ready
      </span>
    );
  }
  if (rendered === 0) {
    return (
      <span className="flex items-center gap-2 text-[var(--text-muted)]">
        <span
          aria-hidden="true"
          className="h-1.5 w-1.5 rounded-full border border-current"
        />
        Not rendered
      </span>
    );
  }
  return (
    <span className="text-[var(--text-muted)]">
      {rendered} of {total} rendered
    </span>
  );
}

export function StudioProjectList({
  projects,
  savedVoices,
  loading,
  error,
  onRetry,
  loadMore,
  emptyAction,
  onOpen,
  onDelete,
}: StudioProjectListProps) {
  const columns: HistoryColumn<StudioProjectSummary>[] = [
    {
      id: "project",
      header: "Project",
      className: "max-w-0",
      cell: (project) => (
        <span className="block truncate font-medium text-[var(--text-primary)]">
          {project.name}
        </span>
      ),
    },
    {
      id: "status",
      header: "Status",
      className: "w-[15rem]",
      cellClassName: "whitespace-nowrap",
      cell: (project) => <ProjectStatus project={project} />,
    },
    {
      id: "length",
      header: "Length",
      className: "w-[6rem]",
      cellClassName: "whitespace-nowrap tabular-nums text-[var(--text-muted)]",
      cell: (project) =>
        project.audio_duration_secs > 0
          ? formatClockDuration(project.audio_duration_secs)
          : "—",
    },
    {
      id: "voice",
      header: "Voice",
      className: "hidden w-[9rem] md:table-cell",
      cellClassName: "truncate text-[var(--text-muted)]",
      cell: (project) =>
        voiceName(
          {
            modelId: project.model_id ?? "",
            voiceMode: project.voice_mode,
            speaker: project.speaker,
            savedVoiceId: project.saved_voice_id,
          },
          savedVoices,
        ),
    },
    {
      id: "edited",
      header: "Edited",
      className: "hidden w-[9rem] sm:table-cell",
      cellClassName: "whitespace-nowrap text-[var(--text-muted)]",
      cell: (project) => formatHistoryDate(project.updated_at),
    },
  ];

  return (
    <HistoryTable
      rows={projects}
      columns={columns}
      getRowId={(project) => project.id}
      getRowLabel={(project) => project.name}
      onOpenRow={(project) => onOpen(project.id)}
      rowActions={(project) => (
        <RowMenu
          label={project.name}
          items={[
            {
              id: "open",
              label: "Open",
              icon: ExternalLink,
              onSelect: () => onOpen(project.id),
            },
            {
              id: "delete",
              label: "Delete",
              icon: Trash2,
              danger: true,
              separatorBefore: true,
              onSelect: () => onDelete(project),
            },
          ]}
        />
      )}
      loading={loading}
      error={error}
      onRetry={onRetry}
      empty={{
        title: "Narrate your first script",
        description:
          "Paste a chapter, an article or a podcast script. Izwi splits it into paragraphs you can edit, listen to and export as one file.",
        action: emptyAction,
      }}
      total={null}
      noun="projects"
      loadMore={loadMore}
    />
  );
}
