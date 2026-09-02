import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { StudioProjectRecord } from "@/api";
import { StudioSegmentEditor } from "@/features/studio/components/StudioSegmentEditor";

function buildProject(segmentCount = 6): StudioProjectRecord {
  return {
    id: "studio-project-1",
    created_at: Date.UTC(2026, 8, 1),
    updated_at: Date.UTC(2026, 8, 2),
    name: "Long-form narration project",
    source_filename: "long-script.txt",
    source_text: "Long script",
    model_id: "tts-model",
    voice_mode: "built_in",
    speaker: "Vivian",
    saved_voice_id: null,
    speed: 1,
    segments: Array.from({ length: segmentCount }, (_, position) => ({
      id: `segment-${position + 1}`,
      project_id: "studio-project-1",
      position,
      text: `Segment ${position + 1} with enough content to exercise a narrow editor layout.`,
      model_id: null,
      voice_mode: null,
      speaker: null,
      saved_voice_id: null,
      input_chars: 1_234 + position,
      speech_record_id: position % 2 === 0 ? `speech-${position + 1}` : null,
      updated_at: Date.UTC(2026, 8, 2),
      generation_time_ms: null,
      audio_duration_secs: position % 2 === 0 ? 12.4 : null,
      audio_filename: null,
    })),
  };
}

describe("StudioSegmentEditor responsive structure", () => {
  it("keeps six-segment metadata and actions in independent wrapping rows", () => {
    const project = buildProject();

    render(
      <StudioSegmentEditor
        project={project}
        segmentDrafts={{}}
        segmentSelections={{}}
        selectedSegmentIdSet={new Set()}
        selectedSegmentCount={0}
        queuedSegmentIdSet={new Set()}
        savingSegmentId={null}
        renderingSegmentId={null}
        addingSegmentAfterSegmentId={null}
        focusSegmentId={null}
        onToggleSelectAll={vi.fn()}
        onRenderSelected={vi.fn()}
        onDeleteSelected={vi.fn()}
        onAddSegment={vi.fn(async () => true)}
        onToggleSegmentSelection={vi.fn()}
        onSaveSegment={vi.fn()}
        onMoveSegment={vi.fn()}
        onMergeSegmentWithNext={vi.fn()}
        onSplitSegment={vi.fn()}
        onRenderSegment={vi.fn()}
        onDeleteSegment={vi.fn()}
        onOpenSegmentSettings={vi.fn()}
        onChangeSegmentDraft={vi.fn()}
        onChangeSegmentCursor={vi.fn()}
        onFocusSegmentHandled={vi.fn()}
        audioUrlForRecordId={(recordId) => `/audio/${recordId}`}
      />,
    );

    expect(screen.getAllByRole("textbox")).toHaveLength(6);
    for (const segment of project.segments) {
      expect(screen.getByTestId(`studio-segment-metadata-${segment.id}`)).toHaveClass(
        "flex-wrap",
      );
      expect(screen.getByTestId(`studio-segment-actions-${segment.id}`)).toHaveClass(
        "w-full",
        "flex-wrap",
      );
      expect(
        screen.getByRole("button", {
          name: `Open settings for segment ${segment.position + 1}`,
        }),
      ).toBeVisible();
      expect(
        screen.getByRole("button", {
          name: `More actions for segment ${segment.position + 1}`,
        }),
      ).toBeVisible();
    }
  });
});
