import { describe, expect, it } from "vitest";

import type { ModelInfo } from "@/api";
import {
  buildTaskRows,
  downloadPlan,
  groupModels,
  hardwareLabel,
  modelDescriptor,
  modelGroup,
  pageChoices,
  readyTasks,
  taskSizeLabels,
  type TaskId,
} from "@/app/onboarding/firstRunPlan";

const GB = 1e9;
const GIB = 1024 ** 3;

function model(
  variant: string,
  task: string,
  overrides: Partial<ModelInfo> = {},
): ModelInfo {
  return {
    variant,
    task,
    status: "not_downloaded",
    device_fit: "fits",
    estimated_size_bytes: GB,
    disk_required_bytes: GB,
    estimated_memory_bytes: 2 * GIB,
    local_path: null,
    size_bytes: null,
    download_progress: null,
    error_message: null,
    ...overrides,
  };
}

const speech = (overrides: Partial<NonNullable<ModelInfo["speech_capabilities"]>>) => ({
  supports_builtin_voices: false,
  built_in_voice_count: null,
  supports_reference_voice: false,
  supports_voice_description: false,
  supports_streaming: true,
  supports_speed_control: false,
  supports_auto_long_form: false,
  ...overrides,
});

const CATALOG = [
  model("Parakeet-TDT-0.6B-v3", "asr", { estimated_size_bytes: 2.5 * GB }),
  model("Qwen3-ASR-0.6B-GGUF", "asr"),
  model("Whisper-Large-v3-Turbo", "asr", { estimated_size_bytes: 1.6 * GB }),
  model("Qwen3.5-4B", "chat", { estimated_size_bytes: 3.4 * GB }),
  model("Qwen3.8-27B-FP8", "chat", {
    device_fit: "too_large",
    estimated_memory_bytes: 80 * GIB,
  }),
  model("Kokoro-82M", "tts", {
    estimated_size_bytes: 0.4 * GB,
    speech_capabilities: speech({ supports_builtin_voices: true, built_in_voice_count: 41 }),
  }),
  model("Voxtral-4B-TTS-2603", "tts", {
    speech_capabilities: speech({ supports_builtin_voices: true, built_in_voice_count: 20 }),
  }),
  model("FishAudio-S2-Pro", "tts", {
    device_fit: "too_large",
    estimated_memory_bytes: 24 * GIB,
    speech_capabilities: speech({ supports_reference_voice: true }),
  }),
  model("Qwen3-TTS-12Hz-0.6B-Base", "tts", {
    estimated_size_bytes: 2.5 * GB,
    speech_capabilities: speech({ supports_reference_voice: true }),
  }),
  model("VibeVoice-1.5B", "tts", {
    speech_capabilities: speech({ supports_reference_voice: true }),
  }),
  model("Qwen3-TTS-12Hz-1.7B-VoiceDesign", "tts", {
    estimated_size_bytes: 4.5 * GB,
    speech_capabilities: speech({ supports_voice_description: true }),
  }),
  model("diar_streaming_sortformer_4spk-v2.1", "diarization"),
  model("Qwen3-ForcedAligner-0.6B", "forced_align"),
  model("LFM2.5-Audio-1.5B-GGUF", "audio_chat", {
    speech_capabilities: speech({ supports_builtin_voices: true, built_in_voice_count: 4 }),
  }),
  model("Qwen3-TTS-Tokenizer-12Hz", "tokenizer"),
];

describe("buildTaskRows", () => {
  it("picks each page's preferred model that fits, reusing Talk's", () => {
    const rows = buildTaskRows(CATALOG);
    const picks = Object.fromEntries(
      rows.map((row) => [row.id, row.models.map((entry) => entry.variant)]),
    );

    expect(picks).toEqual({
      talk: ["Parakeet-TDT-0.6B-v3", "Qwen3.5-4B", "Kokoro-82M"],
      chat: ["Qwen3.5-4B"],
      // Its page prefers Qwen3-ASR, but Talk already downloads Parakeet.
      transcribe: ["Parakeet-TDT-0.6B-v3"],
      read: ["Kokoro-82M"],
      // Fish S2 Pro needs more memory than this device has.
      voices: ["Qwen3-TTS-12Hz-0.6B-Base", "Qwen3-TTS-12Hz-1.7B-VoiceDesign"],
      speakers: [
        "diar_streaming_sortformer_4spk-v2.1",
        "Whisper-Large-v3-Turbo",
        "Qwen3-ForcedAligner-0.6B",
      ],
    });
  });

  it("blocks a task with no model that fits and leaves out one with none listed", () => {
    const rows = buildTaskRows(
      CATALOG.filter((entry) => entry.variant !== "Qwen3-ForcedAligner-0.6B").map(
        (entry) =>
          entry.task === "chat"
            ? { ...entry, device_fit: "too_large" as const, estimated_memory_bytes: 9 * GIB }
            : entry,
      ),
    );

    expect(rows.map((row) => row.id)).not.toContain("speakers");
    const chat = rows.find((row) => row.id === "chat");
    expect(chat?.models).toEqual([]);
    expect(chat?.tooLarge?.variant).toBe("Qwen3.5-4B");
    expect(rows.find((row) => row.id === "talk")?.tooLarge?.variant).toBe(
      "Qwen3.5-4B",
    );
  });
});

describe("taskSizeLabels", () => {
  const rows = buildTaskRows(CATALOG);

  it("counts a model shared by checked tasks once", () => {
    const labels = taskSizeLabels(
      rows,
      new Set<TaskId>(["talk", "chat", "transcribe", "read"]),
    );

    expect(labels.get("talk")).toEqual({ kind: "bytes", bytes: 6.3 * GB });
    expect(labels.get("chat")).toEqual({ kind: "included" });
    expect(labels.get("transcribe")).toEqual({ kind: "included" });
    expect(labels.get("read")).toEqual({ kind: "included" });
    // Unchecked: what checking it would add.
    expect(labels.get("voices")).toEqual({ kind: "bytes", bytes: 7 * GB });
  });

  it("moves a shared model's size to the next checked task", () => {
    const labels = taskSizeLabels(rows, new Set<TaskId>(["chat", "transcribe"]));

    expect(labels.get("chat")).toEqual({ kind: "bytes", bytes: 3.4 * GB });
    expect(labels.get("transcribe")).toEqual({ kind: "bytes", bytes: 2.5 * GB });
    expect(labels.get("talk")).toEqual({ kind: "bytes", bytes: 0.4 * GB });
  });

  it("says when a task's models are already on disk", () => {
    const onDisk = buildTaskRows(
      CATALOG.map((entry) =>
        entry.variant === "Qwen3.5-4B" ? { ...entry, status: "downloaded" as const } : entry,
      ),
    );
    expect(taskSizeLabels(onDisk, new Set<TaskId>(["chat"])).get("chat")).toEqual({
      kind: "downloaded",
    });
  });
});

describe("downloadPlan", () => {
  it("sums what still needs downloading, and disk only when every model says", () => {
    const [parakeet, , whisper] = CATALOG;
    expect(
      downloadPlan([parakeet, whisper, parakeet, { ...whisper, variant: "x", status: "ready" }]),
    ).toMatchObject({ bytes: 4_100_000_000, diskBytes: 2 * GB });
    expect(
      downloadPlan([{ ...parakeet, disk_required_bytes: undefined }]).diskBytes,
    ).toBeNull();
  });
});

describe("modelGroup", () => {
  it("groups by task and speech capabilities, not names", () => {
    const groups = Object.fromEntries(
      CATALOG.map((entry) => [entry.variant, modelGroup(entry)]),
    );
    expect(groups).toMatchObject({
      "Voxtral-4B-TTS-2603": "tts",
      "FishAudio-S2-Pro": "clone",
      "VibeVoice-1.5B": "clone",
      "Qwen3-TTS-12Hz-1.7B-VoiceDesign": "design",
      "Qwen3-ForcedAligner-0.6B": "align",
      "LFM2.5-Audio-1.5B-GGUF": "voice",
      "diar_streaming_sortformer_4spk-v2.1": "speakers",
      "Qwen3-TTS-Tokenizer-12Hz": null,
    });
    // LFM3 Audio speaks too, but its task keeps it with the voice models.
    expect(
      modelGroup(
        model("LFM3-Audio-2B-GGUF", "audio_chat", {
          speech_capabilities: speech({
            supports_builtin_voices: true,
            built_in_voice_count: 2,
          }),
        }),
      ),
    ).toBe("voice");
  });

  it("falls back to route capabilities without a task", () => {
    const routes = {
      openai_chat_completions: false,
      openai_audio_transcriptions: false,
      forced_alignment: true,
    } as NonNullable<ModelInfo["route_capabilities"]>;
    expect(
      modelGroup({ ...model("aligner", "x"), task: undefined, route_capabilities: routes }),
    ).toBe("align");
  });

  it("lists recommended models first and models too large last", () => {
    const chat = groupModels(
      [
        ...CATALOG,
        model("Qwen3.5-2B", "chat", { estimated_size_bytes: 1.9 * GB }),
      ],
      new Set(["Qwen3.5-4B"]),
    ).find((group) => group.id === "chat");
    expect(chat?.models.map((entry) => entry.variant)).toEqual([
      "Qwen3.5-4B",
      "Qwen3.5-2B",
      "Qwen3.8-27B-FP8",
    ]);
  });
});

describe("labels", () => {
  it("describes models by capability and provider", () => {
    const kokoro = CATALOG.find((entry) => entry.variant === "Kokoro-82M")!;
    expect(modelDescriptor(kokoro)).toBe("41 built-in voices · hexgrad");
    expect(
      modelDescriptor({
        ...model("Qwen3.5-4B", "chat"),
        modalities: ["text_input", "image_input"],
        chat_capabilities: {
          supports_thinking: true,
          default_thinking_enabled: true,
          reasoning_efforts: [],
          default_reasoning_effort: null,
          supports_preserve_thinking: false,
        },
      }),
    ).toBe("Reads images, can think before replying · Qwen");
  });

  it("names the memory and backend health reports", () => {
    expect(
      hardwareLabel({
        status: "ok",
        version: "0.1.0",
        runtime: {
          selected_backend: "metal",
          memory: {
            system_total_bytes: 16 * GIB,
            system_available_bytes: null,
            reserved_bytes: 0,
          },
        },
      }),
    ).toBe("Picked for 16 GB of memory · Apple Metal");
    expect(hardwareLabel(null)).toBeNull();
  });
});

describe("pageChoices", () => {
  it("points pages at the models first run downloads", () => {
    const rows = buildTaskRows(CATALOG);
    const talk = rows.find((row) => row.id === "talk")!.models;
    expect(pageChoices(talk)).toEqual([
      ["transcription", "Parakeet-TDT-0.6B-v3"],
      ["chat", "Qwen3.5-4B"],
      ["text-to-speech", "Kokoro-82M"],
    ]);
  });
});

describe("readyTasks", () => {
  const catalog = [
    ...CATALOG,
    model("LFM3-Audio-2B-GGUF", "audio_chat", {
      speech_capabilities: speech({ supports_builtin_voices: true, built_in_voice_count: 2 }),
    }),
  ];
  const rows = buildTaskRows(catalog);
  const talkReadyWith = (variant: string) =>
    readyTasks(
      rows,
      catalog.map((entry) =>
        entry.variant === variant ? { ...entry, status: "downloaded" as const } : entry,
      ),
    ).some((row) => row.id === "talk");

  it("counts Talk ready with only LFM3 Audio downloaded", () => {
    expect(talkReadyWith("LFM3-Audio-2B-GGUF")).toBe(true);
  });

  it("counts Talk ready with only LFM2.5 Audio downloaded", () => {
    expect(talkReadyWith("LFM2.5-Audio-1.5B-GGUF")).toBe(true);
  });

  it("leaves Talk out with neither engine's models downloaded", () => {
    expect(talkReadyWith("Kokoro-82M")).toBe(false);
  });
});
