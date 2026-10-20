import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { StudioProjectSegmentRecord, StudioSegmentRenderState } from "@/api";
import { ScriptParagraph } from "@/features/studio/components/ScriptParagraph";

function segment(render_state: StudioSegmentRenderState, text = "The harbour woke. Gulls circled."): StudioProjectSegmentRecord {
  return {
    id: "segment-1",
    project_id: "project",
    position: 0,
    text,
    model_id: null,
    voice_mode: null,
    speaker: null,
    saved_voice_id: null,
    input_chars: text.length,
    speech_record_id: render_state === "not_rendered" ? null : "take-1",
    updated_at: 0,
    generation_time_ms: null,
    audio_duration_secs: 4,
    audio_filename: null,
    render_fingerprint: null,
    render_state,
    render_error: render_state === "failed" ? "Kokoro couldn't start." : null,
  };
}

function renderParagraph(
  state: StudioSegmentRenderState,
  overrides: Partial<Parameters<typeof ScriptParagraph>[0]> = {},
) {
  const props = {
    segment: segment(state),
    number: 2,
    isFirst: false,
    isLast: false,
    canDelete: true,
    playback: { current: false, playing: false, time: 0 },
    selected: false,
    ownVoiceName: null,
    projectVoiceName: "Heart",
    voiceChoices: [],
    focusRequest: null,
    onFocusHandled: vi.fn(),
    dropIndicator: null,
    onSave: vi.fn().mockResolvedValue(true),
    onSplit: vi.fn(),
    onMergeIntoPrevious: vi.fn(),
    onMergeNext: vi.fn(),
    onMove: vi.fn(),
    onDelete: vi.fn(),
    onRender: vi.fn(),
    onPlay: vi.fn(),
    onPlayFrom: vi.fn(),
    onToggleSelect: vi.fn(),
    onVoice: vi.fn(),
    onDragStart: vi.fn(),
    onDragOver: vi.fn(),
    onDrop: vi.fn(),
    onDragEnd: vi.fn(),
    ...overrides,
  };
  render(<ScriptParagraph {...props} />);
  return props;
}

afterEach(() => {
  vi.useRealTimers();
});

describe("ScriptParagraph", () => {
  it("splits at the caret on Enter", () => {
    const props = renderParagraph("rendered");
    const text = screen.getByRole("textbox", { name: "Paragraph 2" }) as HTMLTextAreaElement;
    text.setSelectionRange(17, 17);
    fireEvent.keyDown(text, { key: "Enter" });
    expect(props.onSplit).toHaveBeenCalledWith("The harbour woke.", "Gulls circled.");
  });

  it("merges into the previous paragraph on Backspace at the start", async () => {
    const props = renderParagraph("rendered");
    const text = screen.getByRole("textbox", { name: "Paragraph 2" }) as HTMLTextAreaElement;
    text.setSelectionRange(0, 0);
    await act(async () => {
      fireEvent.keyDown(text, { key: "Backspace" });
    });
    expect(props.onMergeIntoPrevious).toHaveBeenCalled();
    expect(props.onSave).not.toHaveBeenCalled();
  });

  it("deletes an emptied paragraph on Backspace instead of merging its old text back", async () => {
    const props = renderParagraph("rendered");
    const text = screen.getByRole("textbox", { name: "Paragraph 2" }) as HTMLTextAreaElement;
    fireEvent.change(text, { target: { value: "" } });
    text.setSelectionRange(0, 0);
    await act(async () => {
      fireEvent.keyDown(text, { key: "Backspace" });
    });
    expect(props.onDelete).toHaveBeenCalled();
    expect(props.onMergeIntoPrevious).not.toHaveBeenCalled();
  });

  it("saves edits on its own after a pause and shows them as changed", async () => {
    vi.useFakeTimers();
    const props = renderParagraph("rendered");
    fireEvent.change(screen.getByRole("textbox", { name: "Paragraph 2" }), {
      target: { value: "The harbour woke early. " },
    });
    expect(screen.getByText("Changed")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Render paragraph 2" })).toBeTruthy();
    await act(async () => {
      vi.advanceTimersByTime(800);
    });
    expect(props.onSave).toHaveBeenCalledWith("The harbour woke early.");
  });

  it("plays rendered audio and renders anything else from the gutter", () => {
    const rendered = renderParagraph("rendered");
    fireEvent.click(screen.getByRole("button", { name: "Play paragraph 2" }));
    expect(rendered.onPlay).toHaveBeenCalled();
  });

  it("renders a stale paragraph from the gutter and selects with a modifier", () => {
    const props = renderParagraph("stale");
    const gutter = screen.getByRole("button", { name: "Render paragraph 2" });
    fireEvent.click(gutter);
    expect(props.onRender).toHaveBeenCalled();
    fireEvent.click(gutter, { shiftKey: true });
    expect(props.onToggleSelect).toHaveBeenCalledWith(true);
  });

  it("says why a paragraph failed", () => {
    renderParagraph("failed");
    expect(screen.getByText("Kokoro couldn't start.")).toBeTruthy();
    expect(screen.getByText("Failed")).toBeTruthy();
  });
});
