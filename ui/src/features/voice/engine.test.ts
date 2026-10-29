import { describe, expect, it } from "vitest";

import type { ModelInfo } from "@/api";
import {
  COMPACT_VOICE_MODEL,
  resolveCompactVoiceModel,
  voiceEngineChoosesItsOwnVoice,
  voiceEngineSpeakers,
} from "@/features/voice/engine";

function model(variant: string, status: ModelInfo["status"]): ModelInfo {
  return {
    variant,
    status,
    local_path: null,
    size_bytes: null,
    estimated_size_bytes: 1_000_000_000,
    download_progress: null,
    error_message: null,
  };
}

const lfm25 = "LFM2.5-Audio-1.5B-GGUF";

describe("compact voice model", () => {
  it("defaults to LFM3 Audio", () => {
    expect(COMPACT_VOICE_MODEL).toBe("LFM3-Audio-2B-GGUF");
  });

  it("keeps a chosen model the catalog still offers", () => {
    const models = [model(COMPACT_VOICE_MODEL, "downloaded"), model(lfm25, "not_downloaded")];
    expect(resolveCompactVoiceModel(lfm25, models)).toBe(lfm25);
  });

  it("uses a compact model already on disk before asking for the default", () => {
    const models = [model(COMPACT_VOICE_MODEL, "not_downloaded"), model(lfm25, "downloaded")];
    expect(resolveCompactVoiceModel(null, models)).toBe(lfm25);
    expect(resolveCompactVoiceModel("Kokoro-82M", models)).toBe(lfm25);
  });

  it("prefers the default when it is on disk, and falls back to it otherwise", () => {
    expect(
      resolveCompactVoiceModel(null, [
        model(lfm25, "ready"),
        model(COMPACT_VOICE_MODEL, "downloaded"),
      ]),
    ).toBe(COMPACT_VOICE_MODEL);
    expect(
      resolveCompactVoiceModel(null, [
        model(lfm25, "not_downloaded"),
        model(COMPACT_VOICE_MODEL, "downloaded"),
      ]),
    ).toBe(COMPACT_VOICE_MODEL);
    expect(
      resolveCompactVoiceModel(null, [
        model(lfm25, "not_downloaded"),
        model(COMPACT_VOICE_MODEL, "not_downloaded"),
      ]),
    ).toBe(COMPACT_VOICE_MODEL);
  });
});

describe("voice engine speakers", () => {
  it("offers no voice picker when LFM3 Audio, the default, runs the Compact engine", () => {
    expect(voiceEngineChoosesItsOwnVoice("compact")).toBe(true);
    expect(voiceEngineSpeakers("compact")).toEqual([]);
    expect(voiceEngineSpeakers("compact", "LFM3-Audio-2B-Q8_0-GGUF")).toEqual([]);
  });

  it("keeps LFM2.5 Audio's voices for the Compact engine", () => {
    expect(voiceEngineChoosesItsOwnVoice("compact", lfm25)).toBe(false);
    expect(voiceEngineSpeakers("compact", lfm25).length).toBeGreaterThan(0);
  });
});
