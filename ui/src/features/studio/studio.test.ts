import { describe, expect, it } from "vitest";

import type {
  ModelInfo,
  SavedVoiceSummary,
  StudioProjectRecord,
  StudioProjectSegmentRecord,
  StudioSegmentRenderState,
} from "@/api";
import {
  defaultStudioVoice,
  mainAction,
  paragraphVoice,
  voiceForModel,
  voiceName,
  voiceOptions,
} from "@/features/studio/studio";

function model(variant: string, status: ModelInfo["status"], saved = false): ModelInfo {
  return {
    variant,
    status,
    enabled: true,
    local_path: null,
    size_bytes: null,
    download_progress: null,
    error_message: null,
    speech_capabilities: {
      supports_builtin_voices: !saved,
      supports_reference_voice: saved,
      supports_voice_description: false,
      supports_streaming: true,
      supports_speed_control: true,
      supports_auto_long_form: false,
    },
  } as ModelInfo;
}

const savedVoice = { id: "voice-1", name: "My narrator" } as SavedVoiceSummary;

function segment(
  id: string,
  render_state: StudioSegmentRenderState,
  overrides: Partial<StudioProjectSegmentRecord> = {},
): StudioProjectSegmentRecord {
  return {
    id,
    project_id: "project",
    position: 0,
    text: id,
    model_id: null,
    voice_mode: null,
    speaker: null,
    saved_voice_id: null,
    input_chars: 1,
    speech_record_id: render_state === "not_rendered" ? null : `take-${id}`,
    updated_at: 0,
    generation_time_ms: null,
    audio_duration_secs: render_state === "not_rendered" ? null : 2,
    audio_filename: null,
    render_fingerprint: null,
    render_state,
    render_error: null,
    ...overrides,
  };
}

function project(
  segments: StudioProjectSegmentRecord[],
  activeJob: string[] | null = null,
): StudioProjectRecord {
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
    segments,
    active_render_job: activeJob
      ? {
          id: "job",
          project_id: "project",
          created_at: 0,
          updated_at: 0,
          status: "running",
          error_message: null,
          queued_segment_ids: activeJob,
        }
      : null,
  };
}

describe("mainAction", () => {
  it("renders every paragraph of a new project", () => {
    expect(mainAction(project([segment("a", "not_rendered"), segment("b", "not_rendered")]))).toEqual({
      kind: "render",
      count: 2,
      label: "Render 2 paragraphs",
    });
  });

  it("calls missing, stale and failed paragraphs changes once anything is rendered", () => {
    const action = mainAction(
      project([
        segment("a", "rendered"),
        segment("b", "stale"),
        segment("c", "failed"),
        segment("d", "not_rendered"),
      ]),
    );
    expect(action).toMatchObject({ kind: "render", count: 3, label: "Render 3 changes" });
  });

  it("counts the job's paragraphs while it renders", () => {
    const action = mainAction(
      project(
        [segment("a", "rendered"), segment("b", "rendering"), segment("c", "queued")],
        ["a", "b", "c"],
      ),
    );
    expect(action).toEqual({ kind: "rendering", current: 2, total: 3, jobId: "job" });
  });

  it("offers export when every paragraph is current", () => {
    expect(mainAction(project([segment("a", "rendered")]))).toEqual({ kind: "export" });
  });
});

describe("voices", () => {
  it("lists built-in speakers or saved voices by what the model supports", () => {
    expect(voiceOptions(model("Kokoro-82M", "ready"), [savedVoice])[0]).toMatchObject({
      voiceMode: "built_in",
    });
    expect(voiceOptions(model("Qwen3-TTS-12Hz-0.6B-Base", "ready", true), [savedVoice])).toEqual([
      expect.objectContaining({ voiceMode: "saved", savedVoiceId: "voice-1", name: "My narrator" }),
    ]);
  });

  it("starts new projects on a model that is on this device", () => {
    const voice = defaultStudioVoice(
      [model("Qwen3-TTS-12Hz-0.6B-CustomVoice", "not_downloaded"), model("Kokoro-82M", "downloaded")],
      [],
    );
    expect(voice).toMatchObject({ modelId: "Kokoro-82M", voiceMode: "built_in" });
  });

  it("keeps the chosen voice when the new model has it", () => {
    const kokoro = model("Kokoro-82M", "ready");
    const second = voiceOptions(kokoro, [])[1];
    const voice = voiceForModel(
      { modelId: "Kokoro-82M", voiceMode: "built_in", speaker: second.speaker, savedVoiceId: null },
      kokoro,
      [],
    );
    expect(voice?.speaker).toBe(second.speaker);
  });

  it("names a paragraph's own voice and nothing for one that follows the project", () => {
    const current = project([
      segment("a", "rendered"),
      segment("b", "rendered", { speaker: "am_adam", voice_mode: "built_in" }),
    ]);
    expect(paragraphVoice(current, current.segments[0])).toBeNull();
    const own = paragraphVoice(current, current.segments[1]);
    expect(own).toMatchObject({ modelId: "Kokoro-82M", speaker: "am_adam" });
    expect(voiceName(own!, [])).not.toBe("am_adam");
  });
});
