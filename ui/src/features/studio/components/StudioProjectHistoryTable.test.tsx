import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { StudioProjectSummary } from "@/api";
import { StudioProjectHistoryTable } from "@/features/studio/components/StudioProjectHistoryTable";

function buildProject(overrides: Partial<StudioProjectSummary> = {}): StudioProjectSummary {
  return {
    id: "studio-project-1",
    created_at: Date.UTC(2026, 3, 1, 9, 30),
    updated_at: Date.UTC(2026, 3, 2, 11, 15),
    name: "Narration Project",
    source_filename: "script.txt",
    model_id: "Orpheus-3B-0.1-ft-Q8_0-GGUF",
    voice_mode: "built_in",
    speaker: "Vivian",
    saved_voice_id: null,
    speed: 1,
    segment_count: 12,
    rendered_segment_count: 7,
    stale_segment_count: 0,
    pending_segment_count: 0,
    failed_segment_count: 0,
    audio_duration_secs: 84,
    total_chars: 3200,
    ...overrides,
  };
}

describe("StudioProjectHistoryTable", () => {
  it("opens a project when a row is clicked", () => {
    const onOpenProject = vi.fn();
    render(
      <StudioProjectHistoryTable
        projects={[buildProject()]}
        onOpenProject={onOpenProject}
      />,
    );

    fireEvent.click(
      screen.getByRole("row", { name: /Open Studio project Narration Project/i }),
    );

    expect(onOpenProject).toHaveBeenCalledWith("studio-project-1");
  });

  it("forwards delete requests from row actions", async () => {
    const onRequestDeleteProject = vi.fn();
    render(
      <StudioProjectHistoryTable
        projects={[buildProject()]}
        onOpenProject={vi.fn()}
        onRequestDeleteProject={onRequestDeleteProject}
      />,
    );

    const rowActionsButton = screen.getByRole("button", {
      name: /More actions for Narration Project/i,
    });
    fireEvent.pointerDown(rowActionsButton);
    fireEvent.click(rowActionsButton);
    fireEvent.click(await screen.findByRole("menuitem", { name: /^Delete$/i }));

    expect(onRequestDeleteProject).toHaveBeenCalledWith(
      expect.objectContaining({ id: "studio-project-1", name: "Narration Project" }),
    );
  });

  it("shows empty state with create action", () => {
    const onCreateProject = vi.fn();
    render(
      <StudioProjectHistoryTable
        projects={[]}
        onOpenProject={vi.fn()}
        onCreateProject={onCreateProject}
      />,
    );

    expect(screen.getByText("No Studio projects yet")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /New project/i }));
    expect(onCreateProject).toHaveBeenCalledTimes(1);
  });

  it("renders load-more controls and triggers callback", () => {
    const onLoadMore = vi.fn();
    render(
      <StudioProjectHistoryTable
        projects={[buildProject()]}
        onOpenProject={vi.fn()}
        loadMore={{
          canLoadMore: true,
          loading: false,
          onLoadMore,
        }}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /Load more/i }));
    expect(onLoadMore).toHaveBeenCalledTimes(1);
  });
});
