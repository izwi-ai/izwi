import type {
  SpeechDialogueReference,
  SpeechModelCapabilities,
  SpeechScriptLine,
} from "@/api";

/** One speaker of a scene: its speaker number and the voice it speaks in. */
export interface DialogueSpeaker {
  /** The `<|speaker:N|>` number, 0 to the model's `max_speakers - 1`. */
  speaker: number;
  /** A saved voice with a transcript; null speaks in a voice the seed sets. */
  savedVoiceId: string | null;
}

/** One line of a scene. */
export interface DialogueLine {
  /** Stable key for the row. */
  id: string;
  speaker: number;
  text: string;
}

export interface DialogueScene {
  speakers: DialogueSpeaker[];
  lines: DialogueLine[];
}

/** A dialogue model's limits, from its speech capabilities. */
export interface DialogueLimits {
  maxSpeakers: number;
  maxReferences: number;
}

/**
 * The tags Fish Audio Drama 3 applies to the words they wrap
 * (`<whisper>…</whisper>`, docs/user/features/text-to-speech.md); its point
 * tags (`[laughs]`, …) come from the model's `inline_tags`.
 */
export const FISH_DRAMA3_SCOPED_TAGS: readonly string[] = ["whisper"];

const SPEAKER_TAG = /<\|speaker:(\d+)\|>/g;

let lineCounter = 0;

/** A speaker as the editor names it: its tag number, as the stored text,
 * the API and the CLI number it. */
export function speakerName(speaker: number): string {
  return `Speaker ${speaker}`;
}

export function newDialogueLine(speaker: number, text = ""): DialogueLine {
  lineCounter += 1;
  return { id: `dialogue-line-${lineCounter}`, speaker, text };
}

/** The limits of a model that voices several speakers; null for any other. */
export function dialogueLimits(
  capabilities: SpeechModelCapabilities | null | undefined,
): DialogueLimits | null {
  if (!capabilities?.supports_speaker_dialogue || !capabilities.max_speakers) {
    return null;
  }
  return {
    maxSpeakers: capabilities.max_speakers,
    maxReferences: capabilities.max_references ?? 0,
  };
}

/** A new scene: two speakers in seeded voices, one empty line each. */
export function newDialogueScene(limits: DialogueLimits | null): DialogueScene {
  const count = Math.min(2, limits?.maxSpeakers ?? 2);
  const speakers = Array.from({ length: count }, (_, speaker) => ({
    speaker,
    savedVoiceId: null,
  }));
  return {
    speakers,
    lines: speakers.map(({ speaker }) => newDialogueLine(speaker)),
  };
}

/** The lowest speaker number the scene does not use yet. */
export function nextFreeSpeaker(
  scene: DialogueScene,
  limits: DialogueLimits,
): number | null {
  for (let speaker = 0; speaker < limits.maxSpeakers; speaker += 1) {
    if (!scene.speakers.some((entry) => entry.speaker === speaker)) {
      return speaker;
    }
  }
  return null;
}

/** The request's `script`: each line's speaker and trimmed text. */
export function dialogueScript(lines: DialogueLine[]): SpeechScriptLine[] {
  return lines.map(({ speaker, text }) => ({ speaker, text: text.trim() }));
}

/**
 * The request's `references`: the saved voice of each speaker the lines use,
 * in speaker-list order. Speakers without one speak in seeded voices.
 */
export function dialogueReferences(
  scene: DialogueScene,
  lines: DialogueLine[],
): SpeechDialogueReference[] {
  const used = new Set(lines.map((line) => line.speaker));
  return scene.speakers
    .filter((entry) => entry.savedVoiceId && used.has(entry.speaker))
    .map((entry) => ({
      speaker: entry.speaker,
      saved_voice_id: entry.savedVoiceId as string,
    }));
}

/**
 * What stops the lines from being sent, mirroring the server's checks
 * (`speech_dialogue.rs`): no lines, an empty line, a speaker tag in a line,
 * a speaker past the model's, or more saved voices than it takes. The
 * server still checks everything; its message is shown as it is.
 */
export function dialogueProblem(
  scene: DialogueScene,
  lines: DialogueLine[],
  limits: DialogueLimits,
): string | null {
  if (lines.length === 0) {
    return "Add a line to the scene.";
  }
  for (const [index, line] of lines.entries()) {
    const position = scene.lines.indexOf(line);
    const number = (position >= 0 ? position : index) + 1;
    if (!line.text.trim()) {
      return `Line ${number} is empty. Write it or remove it.`;
    }
    if (line.text.includes("<|speaker")) {
      return `Line ${number} has a speaker tag. Pick the line's speaker instead.`;
    }
    if (line.speaker < 0 || line.speaker >= limits.maxSpeakers) {
      return `Line ${number} names speaker ${line.speaker}; this model's speakers are 0 to ${limits.maxSpeakers - 1}.`;
    }
  }
  const references = dialogueReferences(scene, lines).length;
  if (references > limits.maxReferences) {
    return `This scene uses ${references} saved voices; the model takes at most ${limits.maxReferences}. Give some speakers a seeded voice.`;
  }
  return null;
}

/** Saved voices given to speakers, counted against `max_references`. */
export function assignedVoiceCount(scene: DialogueScene): number {
  return scene.speakers.filter((entry) => entry.savedVoiceId).length;
}

function canonicalSpeaker(digits: string, maxSpeakers: number): number | null {
  if (digits.length > 2 || (digits.length > 1 && digits.startsWith("0"))) {
    return null;
  }
  const speaker = Number(digits);
  return speaker < maxSpeakers ? speaker : null;
}

/**
 * The lines of a stored dialogue: a record's text as the server renders a
 * script (`<|speaker:N|> text` lines). Null unless the text starts with a
 * speaker tag and every tag is one the model takes. Every speaker comes
 * back in a seeded voice; `takeDraft` (take.ts) gives back the voices the
 * record kept.
 */
export function parseDialogueText(
  text: string,
  maxSpeakers: number,
): DialogueScene | null {
  if (!text.trimStart().startsWith("<|speaker:")) {
    return null;
  }
  const tags = [...text.matchAll(SPEAKER_TAG)];
  if (tags.length !== text.split("<|speaker").length - 1) {
    return null;
  }
  const lines: DialogueLine[] = [];
  for (const [index, tag] of tags.entries()) {
    const speaker = canonicalSpeaker(tag[1], maxSpeakers);
    if (speaker === null) {
      return null;
    }
    const start = (tag.index ?? 0) + tag[0].length;
    const end = tags[index + 1]?.index ?? text.length;
    const lineText = text.slice(start, end).trim();
    if (lineText) {
      lines.push(newDialogueLine(speaker, lineText));
    }
  }
  if (lines.length === 0) {
    return null;
  }
  const speakers = [...new Set(lines.map((line) => line.speaker))]
    .sort((a, b) => a - b)
    .map((speaker) => ({ speaker, savedVoiceId: null }));
  return { speakers, lines };
}

/** `tag` put at the caret (or over the selection) of `text`, spaced. */
export function insertTagAt(
  text: string,
  start: number,
  end: number,
  tag: string,
): { text: string; caret: number } {
  const before = text.slice(0, start);
  const after = text.slice(end).replace(/^\s+/, "");
  const lead = before && !/\s$/.test(before) ? " " : "";
  const inserted = `${before}${lead}${tag} `;
  return { text: `${inserted}${after}`, caret: inserted.length };
}

/** The selection of `text` wrapped in `<name>…</name>`; with no selection,
 * an empty pair with the caret inside. */
export function wrapTagAt(
  text: string,
  start: number,
  end: number,
  name: string,
): { text: string; caret: number } {
  const open = `<${name}>`;
  const close = `</${name}>`;
  const selected = text.slice(start, end);
  const wrapped = `${text.slice(0, start)}${open}${selected}`;
  return {
    text: `${wrapped}${close}${text.slice(end)}`,
    caret: selected ? wrapped.length + close.length : wrapped.length,
  };
}
