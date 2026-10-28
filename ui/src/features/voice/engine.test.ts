import { describe, expect, it } from "vitest";

import {
  COMPACT_VOICE_MODEL,
  voiceEngineChoosesItsOwnVoice,
  voiceEngineSpeakers,
} from "@/features/voice/engine";

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
