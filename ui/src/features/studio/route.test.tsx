import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ModelInfo, StudioProjectRecord, StudioProjectSummary } from "@/api";
import { StudioPage } from "./route";

const apiMocks = vi.hoisted(() => ({
  listStudioProjectPage: vi.fn(),
  listSavedVoices: vi.fn(),
  getStudioProject: vi.fn(),
  listStudioProjectPronunciations: vi.fn(),
  createStudioProjectRenderJob: vi.fn(),
  cancelStudioProjectRenderJob: vi.fn(),
  updateStudioProjectSegment: vi.fn(),
  studioProjectSegmentAudioUrl: vi.fn(() => "/audio"),
  deleteStudioProject: vi.fn(),
}));

vi.mock("@/api", () => ({ api: apiMocks }));

const kokoro = {
  variant: "Kokoro-82M",
  status: "downloaded",
  enabled: true,
  speech_capabilities: {
    supports_builtin_voices: true,
    supports_reference_voice: false,
    supports_voice_description: false,
    supports_streaming: true,
    supports_speed_control: true,
    supports_auto_long_form: false,
  },
} as ModelInfo;

const summary: StudioProjectSummary = {
  id: "project-1",
  created_at: 0,
  updated_at: Date.now(),
  name: "Chapter 1",
  source_filename: null,
  model_id: "Kokoro-82M",
  voice_mode: "built_in",
  speaker: "af_heart",
  saved_voice_id: null,
  speed: null,
  segment_count: 2,
  rendered_segment_count: 1,
  stale_segment_count: 1,
  pending_segment_count: 0,
  failed_segment_count: 0,
  audio_duration_secs: 9,
  total_chars: 80,
};

function project(): StudioProjectRecord {
  const segment = (id: string, position: number, text: string, state: "rendered" | "stale") => ({
    id,
    project_id: "project-1",
    position,
    text,
    model_id: null,
    voice_mode: null,
    speaker: null,
    saved_voice_id: null,
    input_chars: text.length,
    speech_record_id: `take-${id}`,
    updated_at: 0,
    generation_time_ms: null,
    audio_duration_secs: 4.5,
    audio_filename: null,
    render_fingerprint: "f",
    render_state: state,
    render_error: null,
  });
  return {
    id: "project-1",
    created_at: 0,
    updated_at: 0,
    name: "Chapter 1",
    source_filename: null,
    source_text: "",
    model_id: "Kokoro-82M",
    voice_mode: "built_in",
    speaker: "af_heart",
    saved_voice_id: null,
    speed: null,
    segments: [
      segment("s1", 0, "The harbour woke.", "rendered"),
      segment("s2", 1, "Gulls circled the breakwater.", "stale"),
    ],
    active_render_job: null,
  };
}

function renderAt(path: string) {
  const onError = vi.fn();
  render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        {["/studio", "/studio/:projectId"].map((routePath) => (
          <Route
            key={routePath}
            path={routePath}
            element={
              <StudioPage
                models={[kokoro]}
                selectedModel={null}
                loading={false}
                downloadProgress={{}}
                onDownload={vi.fn()}
                onLoad={vi.fn()}
                onUnload={vi.fn()}
                onDelete={vi.fn()}
                onSelect={vi.fn()}
                onError={onError}
                onRefresh={vi.fn()}
              />
            }
          />
        ))}
      </Routes>
    </MemoryRouter>,
  );
  return { onError };
}

beforeEach(() => {
  vi.clearAllMocks();
  apiMocks.listSavedVoices.mockResolvedValue([]);
  apiMocks.listStudioProjectPage.mockResolvedValue({
    items: [summary],
    pagination: { next_cursor: null, has_more: false, limit: 24 },
  });
  apiMocks.getStudioProject.mockResolvedValue(project());
  apiMocks.listStudioProjectPronunciations.mockResolvedValue([]);
  apiMocks.createStudioProjectRenderJob.mockResolvedValue({ id: "job" });
});

describe("Studio", () => {
  it("lists projects with where their audio stands", async () => {
    renderAt("/studio");
    expect(await screen.findByText("Chapter 1")).toBeTruthy();
    expect(screen.getByText("1 change to render")).toBeTruthy();
    expect(screen.getByText("0:09")).toBeTruthy();
  });

  it("opens a project as paragraphs and renders its changes", async () => {
    renderAt("/studio/project-1");
    expect(await screen.findByRole("textbox", { name: "Paragraph 1" })).toBeTruthy();
    expect(screen.getByRole("textbox", { name: "Paragraph 2" })).toBeTruthy();
    expect(screen.getByText("2 paragraphs · 0:09 · Saved")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Render 1 change" }));
    await waitFor(() =>
      expect(apiMocks.createStudioProjectRenderJob).toHaveBeenCalledWith("project-1", {
        segment_ids: undefined,
      }),
    );
    expect(apiMocks.getStudioProject).toHaveBeenCalledTimes(2);
  });

  it("says when a project doesn't exist", async () => {
    apiMocks.getStudioProject.mockRejectedValue(new Error("Studio project not found"));
    renderAt("/studio/missing");
    expect(await screen.findByText("This project doesn't exist")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Back to Studio" })).toBeTruthy();
  });
});
