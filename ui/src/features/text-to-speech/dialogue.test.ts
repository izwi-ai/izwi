import { describe, expect, it } from "vitest";

import {
  dialogueLimits,
  dialogueProblem,
  dialogueReferences,
  dialogueScript,
  insertTagAt,
  newDialogueLine,
  parseDialogueText,
  wrapTagAt,
  type DialogueScene,
} from "./dialogue";

const LIMITS = { maxSpeakers: 16, maxReferences: 8 };

function rows(scene: DialogueScene | null) {
  return scene?.lines.map(({ speaker, text }) => ({ speaker, text })) ?? null;
}

describe("dialogue limits", () => {
  it("come from a dialogue model's capabilities and are null for any other", () => {
    expect(
      dialogueLimits({
        supports_builtin_voices: false,
        built_in_voice_count: null,
        supports_reference_voice: true,
        supports_voice_description: false,
        supports_streaming: true,
        supports_speed_control: false,
        supports_auto_long_form: true,
        supports_speaker_dialogue: true,
        max_speakers: 16,
        max_references: 8,
      }),
    ).toEqual(LIMITS);
    expect(
      dialogueLimits({
        supports_builtin_voices: false,
        built_in_voice_count: null,
        supports_reference_voice: true,
        supports_voice_description: false,
        supports_streaming: true,
        supports_speed_control: false,
        supports_auto_long_form: true,
      }),
    ).toBeNull();
    expect(dialogueLimits(null)).toBeNull();
  });
});

describe("parsing a stored dialogue back into rows", () => {
  it("reads the server's rendered script, one row per tagged line", () => {
    const scene = parseDialogueText(
      "<|speaker:0|> Ready when you are.\n<|speaker:3|> Always. [laughs] Let's go.\n<|speaker:0|> <whisper>Quietly</whisper> now.",
      16,
    );
    expect(rows(scene)).toEqual([
      { speaker: 0, text: "Ready when you are." },
      { speaker: 3, text: "Always. [laughs] Let's go." },
      { speaker: 0, text: "<whisper>Quietly</whisper> now." },
    ]);
    // The take keeps no voices: every speaker comes back seeded.
    expect(scene?.speakers).toEqual([
      { speaker: 0, savedVoiceId: null },
      { speaker: 3, savedVoiceId: null },
    ]);
    expect(new Set(scene?.lines.map((line) => line.id)).size).toBe(3);
  });

  it("keeps a line that spans several text lines whole", () => {
    expect(
      rows(parseDialogueText("<|speaker:1|> First,\nthen more.\n<|speaker:2|> Yes.", 16)),
    ).toEqual([
      { speaker: 1, text: "First,\nthen more." },
      { speaker: 2, text: "Yes." },
    ]);
  });

  it("is not a dialogue unless the text starts with a speaker tag", () => {
    expect(parseDialogueText("Hello there.", 16)).toBeNull();
    expect(parseDialogueText("Intro <|speaker:1|> Hi.", 16)).toBeNull();
    expect(parseDialogueText("", 16)).toBeNull();
  });

  it("refuses tags the model would not take", () => {
    expect(parseDialogueText("<|speaker:16|> Too far.", 16)).toBeNull();
    expect(parseDialogueText("<|speaker:01|> Padded.", 16)).toBeNull();
    expect(parseDialogueText("<|speaker:0|> Hi <|speaker:x|> there", 16)).toBeNull();
    expect(parseDialogueText("<|speaker:0|>   ", 16)).toBeNull();
  });
});

describe("dialogue requests", () => {
  const scene: DialogueScene = {
    speakers: [
      { speaker: 0, savedVoiceId: "voice-anna" },
      { speaker: 1, savedVoiceId: null },
      { speaker: 2, savedVoiceId: "voice-ben" },
    ],
    lines: [
      newDialogueLine(0, "  Ready when you are. "),
      newDialogueLine(1, "Always. [laughs]"),
    ],
  };

  it("sends trimmed lines and the saved voices of the speakers they use", () => {
    expect(dialogueScript(scene.lines)).toEqual([
      { speaker: 0, text: "Ready when you are." },
      { speaker: 1, text: "Always. [laughs]" },
    ]);
    // Speaker 1 is seeded and speaker 2 says nothing: only speaker 0's voice.
    expect(dialogueReferences(scene, scene.lines)).toEqual([
      { speaker: 0, saved_voice_id: "voice-anna" },
    ]);
    expect(dialogueReferences(scene, [scene.lines[1]])).toEqual([]);
  });

  it("mirrors the server's checks before sending", () => {
    expect(dialogueProblem(scene, scene.lines, LIMITS)).toBeNull();
    expect(dialogueProblem(scene, [], LIMITS)).toBe("Add a line to the scene.");

    const blank = { ...scene, lines: [...scene.lines, newDialogueLine(2, "  ")] };
    expect(dialogueProblem(blank, blank.lines, LIMITS)).toMatch(/^Line 3 is empty/);
    // A preview of that line alone still names it by its place in the scene.
    expect(dialogueProblem(blank, [blank.lines[2]], LIMITS)).toMatch(/^Line 3 is empty/);

    const tagged = { ...scene, lines: [newDialogueLine(0, "<|speaker:1|> Hi")] };
    expect(dialogueProblem(tagged, tagged.lines, LIMITS)).toMatch(/speaker tag/);

    const crowded = {
      speakers: scene.speakers,
      lines: [newDialogueLine(0, "One."), newDialogueLine(2, "Two.")],
    };
    expect(
      dialogueProblem(crowded, crowded.lines, { maxSpeakers: 16, maxReferences: 1 }),
    ).toMatch(/uses 2 saved voices; the model takes at most 1/);
    expect(
      dialogueProblem(crowded, crowded.lines, { maxSpeakers: 2, maxReferences: 8 }),
    ).toMatch(/names speaker 2; this model's speakers are 0 to 1/);
  });
});

describe("tag helpers", () => {
  it("puts a point tag at the caret with spaces around it", () => {
    expect(insertTagAt("Hello there", 5, 5, "[laughs]")).toEqual({
      text: "Hello [laughs] there",
      caret: 15,
    });
    expect(insertTagAt("", 0, 0, "[sighs]")).toEqual({
      text: "[sighs] ",
      caret: 8,
    });
  });

  it("wraps the selection in a scoped tag, or leaves the caret inside", () => {
    expect(wrapTagAt("say it quietly", 7, 14, "whisper")).toEqual({
      text: "say it <whisper>quietly</whisper>",
      caret: 33,
    });
    expect(wrapTagAt("Now ", 4, 4, "whisper")).toEqual({
      text: "Now <whisper></whisper>",
      caret: 13,
    });
  });
});
