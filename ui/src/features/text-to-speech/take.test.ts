import { describe, expect, it } from "vitest";

import type { SpeechHistoryRecord } from "@/api";
import { takeDraft } from "./take";

const SCENE =
  "<|speaker:0|> Ready when you are.\n<|speaker:1|> Always.\n<|speaker:3|> Me too.";

function take(
  overrides: Partial<SpeechHistoryRecord> = {},
): Pick<SpeechHistoryRecord, "id" | "model_id" | "input_text" | "references" | "seed"> {
  return {
    id: "take-1",
    model_id: "FishAudio-Drama-3",
    input_text: SCENE,
    ...overrides,
  };
}

function voices(draft: ReturnType<typeof takeDraft>) {
  return draft.dialogue?.speakers.map(({ speaker, savedVoiceId }) => ({
    speaker,
    savedVoiceId,
  }));
}

describe("a take reopened in the generation dialog", () => {
  it("gives each speaker back its saved voice, and the seed", () => {
    const draft = takeDraft(
      take({
        references: [
          { speaker: 3, saved_voice_id: "voice-ben" },
          { speaker: 0, saved_voice_id: "voice-anna" },
        ],
        seed: 42,
      }),
      16,
      true,
    );
    expect(draft.editsTakeId).toBe("take-1");
    expect(draft.modelId).toBe("FishAudio-Drama-3");
    expect(draft.dialogue?.lines.map(({ speaker, text }) => ({ speaker, text }))).toEqual([
      { speaker: 0, text: "Ready when you are." },
      { speaker: 1, text: "Always." },
      { speaker: 3, text: "Me too." },
    ]);
    // Speaker 1 had no reference: a seeded voice, which the seed brings back.
    expect(voices(draft)).toEqual([
      { speaker: 0, savedVoiceId: "voice-anna" },
      { speaker: 1, savedVoiceId: null },
      { speaker: 3, savedVoiceId: "voice-ben" },
    ]);
    expect(draft.seed).toBe("42");
    expect(draft.text).toBe("");
    expect(draft.note).toBeNull();
  });

  it("gives back a speaker's saved voice even beside a clip of it", () => {
    const draft = takeDraft(
      take({
        references: [
          { speaker: 0 },
          { speaker: 0, saved_voice_id: "voice-anna" },
        ],
      }),
      16,
      true,
    );
    expect(voices(draft)?.[0]).toEqual({
      speaker: 0,
      savedVoiceId: "voice-anna",
    });
    expect(draft.note).toBeNull();
  });

  it("starts a speaker whose voice was a clip in a seeded voice, and says so", () => {
    const draft = takeDraft(
      take({
        references: [
          { speaker: 0, saved_voice_id: "voice-anna" },
          { speaker: 1 },
          { speaker: 3 },
        ],
      }),
      16,
      false,
    );
    expect(draft.editsTakeId).toBeNull();
    expect(voices(draft)).toEqual([
      { speaker: 0, savedVoiceId: "voice-anna" },
      { speaker: 1, savedVoiceId: null },
      { speaker: 3, savedVoiceId: null },
    ]);
    expect(draft.note).toBe(
      "Speaker 1 and Speaker 3 spoke in reference clips, which takes do not keep, so they start in seeded voices.",
    );
    // No seed in the request: a new one, as a new dialog has.
    expect(draft.seed).toBeNull();

    expect(
      takeDraft(take({ references: [{ speaker: 1 }] }), 16, false).note,
    ).toBe(
      "Speaker 1 spoke in a reference clip, which takes do not keep, so they start in a seeded voice.",
    );
  });

  it("starts every speaker of a take made before takes kept voices seeded", () => {
    const draft = takeDraft(take(), 16, false);
    expect(voices(draft)?.every((entry) => entry.savedVoiceId === null)).toBe(true);
    expect(draft.note).toMatch(/made before takes kept their voices and seed/);
    expect(draft.seed).toBeNull();
  });

  it("says nothing when every speaker spoke in a seeded voice", () => {
    expect(takeDraft(take({ references: [], seed: 7 }), 16, true).note).toBeNull();
  });

  it("gives a single-voice take its text and speaker 0's saved voice", () => {
    const draft = takeDraft(
      take({
        input_text: "Just one voice.",
        references: [{ speaker: 0, saved_voice_id: "voice-anna" }],
        seed: 5,
      }),
      16,
      true,
    );
    expect(draft.dialogue).toBeNull();
    expect(draft.text).toBe("Just one voice.");
    expect(draft.savedVoiceId).toBe("voice-anna");
    expect(draft.seed).toBe("5");
    expect(draft.note).toBeNull();

    const clip = takeDraft(
      take({ input_text: "Just one voice.", references: [{ speaker: 0 }] }),
      16,
      true,
    );
    expect(clip.savedVoiceId).toBeNull();
    expect(clip.note).toBe(
      "This take spoke in a reference clip, which takes do not keep, so it starts in a seeded voice.",
    );
  });

  it("reads a script as text for a model without dialogue", () => {
    const draft = takeDraft(take(), null, true);
    expect(draft.dialogue).toBeNull();
    expect(draft.text).toBe(SCENE);
  });

  it("leaves the seed empty when it is past what JavaScript holds exactly", () => {
    // 2^64 - 1 as JSON.parse reads it: rounded.
    const seed = JSON.parse("18446744073709551615") as number;
    expect(takeDraft(take({ references: [], seed }), 16, true).seed).toBe("");
  });
});
