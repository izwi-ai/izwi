import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { StudioProjectRecord } from "@/api";

const apiMocks = vi.hoisted(() => ({
  downloadAudioFile: vi.fn(),
  studioProjectAudioUrl: vi.fn(
    (_id: string, options: { pause_ms?: number }) =>
      `http://localhost:8080/v1/studio/project-1/audio?pause_ms=${options.pause_ms}`,
  ),
}));
const saveMocks = vi.hoisted(() => ({
  saveFile: vi.fn(),
  savesWithDialog: vi.fn(() => true),
}));

vi.mock("@/api", () => ({ api: apiMocks }));
vi.mock("@/shared/saveFile", () => saveMocks);

import { StudioExportDialog } from "./StudioExportDialog";

function project(): StudioProjectRecord {
  const segment = (id: string, position: number, text: string) => ({
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
    render_state: "rendered" as const,
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
      segment("s1", 0, "The harbour woke."),
      segment("s2", 1, "Gulls circled the breakwater."),
    ],
    active_render_job: null,
  };
}

function renderDialog() {
  const onClose = vi.fn();
  render(
    <StudioExportDialog
      open
      onClose={onClose}
      project={project()}
      onRender={vi.fn(async () => true)}
    />,
  );
  return { onClose };
}

describe("StudioExportDialog", () => {
  beforeEach(() => {
    apiMocks.downloadAudioFile.mockReset();
    saveMocks.saveFile.mockReset();
  });

  it("saves the audio with the chosen pause, then the script", async () => {
    apiMocks.downloadAudioFile.mockResolvedValue(true);
    saveMocks.saveFile.mockResolvedValue(true);
    const { onClose } = renderDialog();

    expect(screen.getByText("You choose where it's saved.")).toBeTruthy();
    fireEvent.click(screen.getByRole("radio", { name: "1 s" }));
    fireEvent.click(screen.getByRole("switch", { name: /Include script/ }));
    fireEvent.click(screen.getByRole("button", { name: /Export…/ }));

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(apiMocks.downloadAudioFile).toHaveBeenCalledWith(
      "http://localhost:8080/v1/studio/project-1/audio?pause_ms=1000",
      "chapter-1.wav",
    );
    expect(saveMocks.saveFile).toHaveBeenCalledWith(
      "chapter-1-script.txt",
      "The harbour woke.\n\nGulls circled the breakwater.",
    );
  });

  it("stays open, saving nothing else, when the save dialog is cancelled", async () => {
    apiMocks.downloadAudioFile.mockResolvedValue(false);
    const { onClose } = renderDialog();

    fireEvent.click(screen.getByRole("switch", { name: /Include script/ }));
    fireEvent.click(screen.getByRole("button", { name: /Export…/ }));

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /Export…/ }).hasAttribute("disabled"),
      ).toBe(false),
    );
    expect(apiMocks.downloadAudioFile).toHaveBeenCalledTimes(1);
    expect(saveMocks.saveFile).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it("shows the server's reason when the export fails", async () => {
    apiMocks.downloadAudioFile.mockRejectedValue(
      new Error("Render these paragraphs first"),
    );
    renderDialog();

    fireEvent.click(screen.getByRole("button", { name: /Export…/ }));

    expect((await screen.findByRole("alert")).textContent).toContain(
      "Render these paragraphs first",
    );
  });
});
