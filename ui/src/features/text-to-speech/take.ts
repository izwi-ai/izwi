import type { SpeechHistoryRecord, SpeechRecordReference } from "@/api";
import {
  parseDialogueText,
  speakerName,
  type DialogueScene,
} from "@/features/text-to-speech/dialogue";

/** A take reopened in the generation dialog: its text, voices and seed as
 * its record keeps them. */
export interface TakeDraft {
  /** The take a new one continues from (`previous_take_ids`): set by Edit
   * this take, null when the take only fills in a fresh one. */
  editsTakeId: string | null;
  /** The take's model; the take is edited only while it is selected. */
  modelId: string;
  /** Rows, when the text is a script the model takes. */
  dialogue: DialogueScene | null;
  /** The single-voice text: the take's, unless it is a dialogue. */
  text: string;
  /** The single-voice saved voice: speaker 0's. */
  savedVoiceId: string | null;
  /** The seed field's value: "" for a seed this page cannot hold exactly,
   * null for a new random one (the take had none). */
  seed: string | null;
  /** Which voices could not come back, said to the user; null when all did. */
  note: string | null;
}

type TakeRecord = Pick<
  SpeechHistoryRecord,
  "id" | "model_id" | "input_text" | "references" | "seed"
>;

/** "Speaker 1", "Speaker 1 and Speaker 3", "Speaker 0, Speaker 1 and …". */
function speakerList(speakers: number[]): string {
  const names = speakers.map(speakerName);
  return names.length > 1
    ? `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`
    : (names[0] ?? "");
}

/**
 * What the dialog says when a take's voices do not all come back. Saved
 * voices and the seed do; a clip's audio is not kept, and a take made
 * before takes kept their voices (no `references`) kept none.
 */
function lostVoicesNote(
  references: SpeechRecordReference[] | undefined,
  speakers: number[],
  dialogue: boolean,
): string | null {
  if (references === undefined) {
    return dialogue
      ? "This take was made before takes kept their voices and seed, so every speaker starts in a seeded voice: pick their voices again."
      : "This take was made before takes kept their voice and seed, so it starts in a seeded voice: pick its voice again.";
  }
  // A speaker voiced by a saved voice comes back even beside a clip.
  const clips = speakers.filter(
    (speaker) =>
      references.some(
        (reference) =>
          reference.speaker === speaker && !reference.saved_voice_id,
      ) &&
      !references.some(
        (reference) =>
          reference.speaker === speaker && reference.saved_voice_id,
      ),
  );
  if (clips.length === 0) {
    return null;
  }
  if (!dialogue) {
    return "This take spoke in a reference clip, which takes do not keep, so it starts in a seeded voice.";
  }
  return clips.length === 1
    ? `${speakerList(clips)} spoke in a reference clip, which takes do not keep, so they start in a seeded voice.`
    : `${speakerList(clips)} spoke in reference clips, which takes do not keep, so they start in seeded voices.`;
}

/**
 * A take as the generation dialog reopens it: its lines (rows when its text
 * parses as a dialogue, else the text), the saved voice of each speaker
 * from `references`, and its seed when it is a safe integer. A speaker
 * whose voice was a clip, and every speaker of a take made before takes
 * kept their voices, starts in a seeded voice.
 */
export function takeDraft(
  record: TakeRecord,
  maxSpeakers: number | null,
  edit: boolean,
): TakeDraft {
  const references = record.references;
  const savedVoice = (speaker: number) =>
    references?.find(
      (reference) => reference.speaker === speaker && reference.saved_voice_id,
    )?.saved_voice_id ?? null;
  const parsed = maxSpeakers
    ? parseDialogueText(record.input_text, maxSpeakers)
    : null;
  const dialogue = parsed
    ? {
        ...parsed,
        speakers: parsed.speakers.map((entry) => ({
          ...entry,
          savedVoiceId: savedVoice(entry.speaker),
        })),
      }
    : null;
  const seed = record.seed;
  return {
    editsTakeId: edit ? record.id : null,
    modelId: record.model_id ?? "",
    dialogue,
    text: dialogue ? "" : record.input_text,
    savedVoiceId: dialogue ? null : savedVoice(0),
    seed:
      seed == null
        ? null
        : Number.isSafeInteger(seed) && seed >= 0
          ? String(seed)
          : "",
    note: lostVoicesNote(
      references,
      dialogue ? dialogue.speakers.map((entry) => entry.speaker) : [0],
      dialogue !== null,
    ),
  };
}
