import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { StudioProjectRecord, StudioProjectSegmentRecord } from "@/api";
import { useStudioProject } from "./useStudioProject";

const apiMocks = vi.hoisted(() => ({
  getStudioProject: vi.fn(),
  listStudioProjectPronunciations: vi.fn(),
}));

vi.mock("@/api", () => ({ api: apiMocks }));

function paragraph(text: string, updatedAt: number): StudioProjectSegmentRecord {
  return {
    id: "p1",
    project_id: "project",
    position: 0,
    text,
    model_id: null,
    voice_mode: null,
    speaker: null,
    saved_voice_id: null,
    input_chars: text.length,
    speech_record_id: null,
    updated_at: updatedAt,
    generation_time_ms: null,
    audio_duration_secs: null,
    audio_filename: null,
    render_fingerprint: null,
    render_state: "not_rendered",
    render_error: null,
  };
}

function project(segment: StudioProjectSegmentRecord): StudioProjectRecord {
  return {
    id: "project",
    created_at: 0,
    updated_at: 0,
    name: "Harbour",
    source_filename: null,
    source_text: "",
    model_id: "Kokoro-82M",
    voice_mode: "built_in",
    speaker: "af_heart",
    saved_voice_id: null,
    speed: null,
    segments: [segment],
    active_render_job: null,
  };
}

describe("useStudioProject", () => {
  beforeEach(() => {
    apiMocks.getStudioProject.mockResolvedValue(project(paragraph("The harbour", 1)));
    apiMocks.listStudioProjectPronunciations.mockResolvedValue([]);
  });

  it("keeps a newer paragraph when an older reply arrives last", async () => {
    const { result } = renderHook(() => useStudioProject("project", vi.fn()));
    await waitFor(() => expect(result.current.project).not.toBeNull());

    let resolveFirst!: (value: StudioProjectRecord) => void;
    const first = new Promise<StudioProjectRecord>((resolve) => {
      resolveFirst = resolve;
    });
    await act(async () => {
      const older = result.current.mutate(() => first, "save failed");
      await result.current.mutate(
        async () => project(paragraph("The harbour woke.", 3)),
        "save failed",
      );
      resolveFirst(project(paragraph("The harbour woke", 2)));
      await older;
    });

    expect(result.current.project?.segments[0].text).toBe("The harbour woke.");
  });
});
