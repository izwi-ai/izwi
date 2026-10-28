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

describe("compact voice model", () => {
  const lfm3 = "LFM3-Audio-2B-GGUF";

  it("keeps a chosen model the catalog still offers", () => {
    const models = [model(COMPACT_VOICE_MODEL, "downloaded"), model(lfm3, "not_downloaded")];
    expect(resolveCompactVoiceModel(lfm3, models)).toBe(lfm3);
  });

  it("uses a compact model already on disk before asking for the default", () => {
    const models = [model(COMPACT_VOICE_MODEL, "not_downloaded"), model(lfm3, "downloaded")];
    expect(resolveCompactVoiceModel(null, models)).toBe(lfm3);
    expect(resolveCompactVoiceModel("Kokoro-82M", models)).toBe(lfm3);
  });

  it("prefers the default when it is on disk, and falls back to it otherwise", () => {
    expect(
      resolveCompactVoiceModel(null, [
        model(lfm3, "ready"),
        model(COMPACT_VOICE_MODEL, "downloaded"),
      ]),
    ).toBe(COMPACT_VOICE_MODEL);
    expect(
      resolveCompactVoiceModel(null, [
        model(lfm3, "not_downloaded"),
        model(COMPACT_VOICE_MODEL, "not_downloaded"),
      ]),
    ).toBe(COMPACT_VOICE_MODEL);
  });
});

describe("voice engine speakers", () => {
  it("offers no voice picker when LFM3 Audio runs the Compact engine", () => {
    expect(voiceEngineChoosesItsOwnVoice("compact", "LFM3-Audio-2B-GGUF")).toBe(true);
    expect(voiceEngineSpeakers("compact", "LFM3-Audio-2B-Q8_0-GGUF")).toEqual([]);
  });

  it("keeps LFM2.5 Audio's voices for the Compact engine", () => {
    expect(voiceEngineChoosesItsOwnVoice("compact", COMPACT_VOICE_MODEL)).toBe(false);
    expect(voiceEngineSpeakers("compact", COMPACT_VOICE_MODEL).length).toBeGreaterThan(0);
  });
});
