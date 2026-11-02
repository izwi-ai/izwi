import { useId, useRef } from "react";
import { Loader2, Pause, Play, Plus, Trash2 } from "lucide-react";

import type { SavedVoiceSummary } from "@/api";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  assignedVoiceCount,
  insertTagAt,
  newDialogueLine,
  nextFreeSpeaker,
  speakerName,
  wrapTagAt,
  type DialogueLimits,
  type DialogueLine,
  type DialogueScene,
} from "@/features/text-to-speech/dialogue";
import { savedVoiceLabel } from "@/features/text-to-speech/support";

/** The voice picker's "no saved voice" choice (Radix Select takes no empty
 * value). */
const SEEDED_VOICE = "__seeded__";

const TRIGGER_CLASS =
  "h-8 rounded-[8px] border-(--border-default) bg-(--surface-panel) px-2.5 text-[13px] shadow-none";

const SCOPED_TAG_PREFIX = "scoped:";

interface DialogueEditorProps {
  scene: DialogueScene;
  onChange: (scene: DialogueScene) => void;
  limits: DialogueLimits;
  savedVoices: SavedVoiceSummary[];
  savedVoicesLoading?: boolean;
  savedVoicesError?: string | null;
  /** Tags placed where they stand (`[laughs]`), from the model. */
  pointTags: string[];
  /** Tags that wrap words (`whisper`), from the model's family. */
  scopedTags: readonly string[];
  disabled?: boolean;
  /** The line whose preview is being generated. */
  previewingLineId: string | null;
  /** The line whose preview is playing. */
  playingLineId: string | null;
  onPreviewLine: (line: DialogueLine) => void;
}

/**
 * The dialogue editor of the text-to-speech dialog: speakers, each in a
 * saved voice with a transcript or a seeded voice, and the lines they say.
 */
export function DialogueEditor({
  scene,
  onChange,
  limits,
  savedVoices,
  savedVoicesLoading = false,
  savedVoicesError = null,
  pointTags,
  scopedTags,
  disabled = false,
  previewingLineId,
  playingLineId,
  onPreviewLine,
}: DialogueEditorProps) {
  const baseId = useId();
  const textareas = useRef(new Map<string, HTMLTextAreaElement>());
  const assigned = assignedVoiceCount(scene);
  const referencesFull = assigned >= limits.maxReferences;
  const freeSpeaker = nextFreeSpeaker(scene, limits);
  const voiceName = (savedVoiceId: string | null) => {
    if (!savedVoiceId) {
      return "seeded voice";
    }
    const voice = savedVoices.find((entry) => entry.id === savedVoiceId);
    return voice ? voice.name : "saved voice";
  };

  const updateLine = (id: string, patch: Partial<DialogueLine>) =>
    onChange({
      ...scene,
      lines: scene.lines.map((line) =>
        line.id === id ? { ...line, ...patch } : line,
      ),
    });

  const focusLine = (id: string, caret?: number) =>
    requestAnimationFrame(() => {
      const field = textareas.current.get(id);
      field?.focus();
      if (field && caret !== undefined) {
        field.setSelectionRange(caret, caret);
      }
    });

  const addLine = () => {
    const last = scene.lines[scene.lines.length - 1];
    const order = scene.speakers.map((entry) => entry.speaker);
    // Alternate: the speaker after the last line's.
    const lastIndex = last ? order.indexOf(last.speaker) : -1;
    const speaker = order[(lastIndex + 1) % order.length] ?? 0;
    const line = newDialogueLine(speaker);
    onChange({ ...scene, lines: [...scene.lines, line] });
    focusLine(line.id);
  };

  const insertTag = (line: DialogueLine, value: string) => {
    const field = textareas.current.get(line.id);
    const start = field?.selectionStart ?? line.text.length;
    const end = field?.selectionEnd ?? line.text.length;
    const next = value.startsWith(SCOPED_TAG_PREFIX)
      ? wrapTagAt(line.text, start, end, value.slice(SCOPED_TAG_PREFIX.length))
      : insertTagAt(line.text, start, end, value);
    updateLine(line.id, { text: next.text });
    focusLine(line.id, next.caret);
  };

  const hasTags = pointTags.length > 0 || scopedTags.length > 0;

  return (
    <div className="flex flex-col gap-5">
      <section
        aria-labelledby={`${baseId}-speakers`}
        className="flex flex-col gap-2"
      >
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h3
            id={`${baseId}-speakers`}
            className="text-[13.5px] font-medium text-(--text-primary)"
          >
            Speakers
          </h3>
          <span className="text-xs text-(--text-subtle)">
            {scene.speakers.length} of {limits.maxSpeakers} speakers ·{" "}
            {assigned} of {limits.maxReferences} saved voices
          </span>
        </div>
        <ul className="flex flex-col gap-1.5">
          {scene.speakers.map((entry) => {
            const name = speakerName(entry.speaker);
            const lineCount = scene.lines.filter(
              (line) => line.speaker === entry.speaker,
            ).length;
            const usageId = `${baseId}-speaker-${entry.speaker}-usage`;
            return (
              <li
                key={entry.speaker}
                className="grid grid-cols-[6.5rem_minmax(0,1fr)_auto] items-center gap-2"
              >
                <span className="flex min-w-0 flex-col">
                  <span className="text-[13px] font-medium text-(--text-primary)">
                    {name}
                  </span>
                  <span id={usageId} className="text-[11.5px] text-(--text-subtle)">
                    {lineCount === 1 ? "1 line" : `${lineCount} lines`}
                  </span>
                </span>
                <Select
                  value={entry.savedVoiceId ?? SEEDED_VOICE}
                  onValueChange={(value) =>
                    onChange({
                      ...scene,
                      speakers: scene.speakers.map((speaker) =>
                        speaker.speaker === entry.speaker
                          ? {
                              ...speaker,
                              savedVoiceId: value === SEEDED_VOICE ? null : value,
                            }
                          : speaker,
                      ),
                    })
                  }
                  disabled={disabled || savedVoicesLoading}
                >
                  <SelectTrigger
                    aria-label={`${name} voice`}
                    className={TRIGGER_CLASS}
                  >
                    <SelectValue placeholder="Seeded voice" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={SEEDED_VOICE}>Seeded voice</SelectItem>
                    {savedVoices.map((voice) => {
                      const noTranscript = voice.reference_text_chars <= 0;
                      // A speaker that has a saved voice may swap it.
                      const overLimit = referencesFull && !entry.savedVoiceId;
                      return (
                        <SelectItem
                          key={voice.id}
                          value={voice.id}
                          disabled={noTranscript || overLimit}
                        >
                          {savedVoiceLabel(voice)}
                          {noTranscript ? " · no transcript" : ""}
                        </SelectItem>
                      );
                    })}
                  </SelectContent>
                </Select>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  aria-label={`Remove ${name}`}
                  aria-describedby={usageId}
                  disabled={disabled || lineCount > 0 || scene.speakers.length <= 1}
                  onClick={() =>
                    onChange({
                      ...scene,
                      speakers: scene.speakers.filter(
                        (speaker) => speaker.speaker !== entry.speaker,
                      ),
                    })
                  }
                >
                  <Trash2 aria-hidden="true" />
                </Button>
              </li>
            );
          })}
        </ul>
        <p className="text-xs leading-4 text-(--text-subtle)">
          {savedVoicesError ? (
            <span className="text-(--danger-text)">{savedVoicesError}</span>
          ) : savedVoicesLoading ? (
            "Loading saved voices…"
          ) : referencesFull ? (
            `A scene takes at most ${limits.maxReferences} saved voices; other speakers speak in voices the seed sets.`
          ) : (
            "A saved voice needs its transcript. A speaker without one speaks in a voice the seed sets."
          )}
        </p>
        <div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-8 gap-1.5"
            onClick={() =>
              freeSpeaker !== null &&
              onChange({
                ...scene,
                speakers: [
                  ...scene.speakers,
                  { speaker: freeSpeaker, savedVoiceId: null },
                ],
              })
            }
            disabled={disabled || freeSpeaker === null}
          >
            <Plus aria-hidden="true" />
            Add speaker
          </Button>
        </div>
      </section>

      <section
        aria-labelledby={`${baseId}-lines`}
        className="flex flex-col gap-2"
      >
        <h3
          id={`${baseId}-lines`}
          className="text-[13.5px] font-medium text-(--text-primary)"
        >
          Lines
        </h3>
        <ol className="flex flex-col gap-2">
          {scene.lines.map((line, index) => {
            const number = index + 1;
            const previewing = previewingLineId === line.id;
            const playing = playingLineId === line.id;
            return (
              <li
                key={line.id}
                aria-label={`Line ${number}`}
                className="flex flex-col gap-2 rounded-[12px] border border-(--border-subtle) bg-(--surface-panel) p-2.5"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <Select
                    value={String(line.speaker)}
                    onValueChange={(value) =>
                      updateLine(line.id, { speaker: Number(value) })
                    }
                    disabled={disabled}
                  >
                    <SelectTrigger
                      aria-label={`Line ${number} speaker`}
                      className={`${TRIGGER_CLASS} w-fit gap-2`}
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {scene.speakers.map((entry) => (
                        <SelectItem
                          key={entry.speaker}
                          value={String(entry.speaker)}
                        >
                          {speakerName(entry.speaker)} ·{" "}
                          {voiceName(entry.savedVoiceId)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {hasTags ? (
                    <Select
                      value=""
                      onValueChange={(value) => insertTag(line, value)}
                      disabled={disabled}
                    >
                      <SelectTrigger
                        aria-label={`Insert a tag into line ${number}`}
                        className={`${TRIGGER_CLASS} w-fit gap-2 text-xs`}
                      >
                        <SelectValue placeholder="Insert a tag" />
                      </SelectTrigger>
                      <SelectContent>
                        {pointTags.map((tag) => (
                          <SelectItem key={tag} value={tag}>
                            {tag}
                          </SelectItem>
                        ))}
                        {scopedTags.map((tag) => (
                          <SelectItem
                            key={tag}
                            value={`${SCOPED_TAG_PREFIX}${tag}`}
                          >
                            {`<${tag}>…</${tag}>`}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  ) : null}
                  {/* The line's actions wrap together, never apart. */}
                  <div className="ml-auto flex items-center gap-1">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="h-8 gap-1.5"
                      aria-label={
                        previewing
                          ? `Generating a preview of line ${number}`
                          : playing
                            ? `Stop the preview of line ${number}`
                            : `Preview line ${number}`
                      }
                      disabled={
                        disabled || (previewingLineId !== null && !previewing)
                      }
                      aria-busy={previewing}
                      onClick={() => {
                        if (!previewing) {
                          onPreviewLine(line);
                        }
                      }}
                    >
                      {previewing ? (
                        <Loader2 aria-hidden="true" className="animate-spin" />
                      ) : playing ? (
                        <Pause aria-hidden="true" />
                      ) : (
                        <Play aria-hidden="true" />
                      )}
                      {playing ? "Stop" : "Preview"}
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      aria-label={`Remove line ${number}`}
                      disabled={disabled}
                      onClick={() =>
                        onChange({
                          ...scene,
                          lines: scene.lines.filter((entry) => entry.id !== line.id),
                        })
                      }
                    >
                      <Trash2 aria-hidden="true" />
                    </Button>
                  </div>
                </div>
                <textarea
                  ref={(node) => {
                    if (node) {
                      textareas.current.set(line.id, node);
                    } else {
                      textareas.current.delete(line.id);
                    }
                  }}
                  aria-label={`Line ${number} text`}
                  value={line.text}
                  onChange={(event) =>
                    updateLine(line.id, { text: event.target.value })
                  }
                  disabled={disabled}
                  rows={2}
                  placeholder={`What ${speakerName(line.speaker)} says…`}
                  className="w-full resize-y rounded-[10px] border border-(--border-default) bg-(--bg-surface-0) px-3 py-2 text-[14px] leading-6 text-(--text-primary) outline-hidden transition-colors placeholder:text-(--text-subtle) focus:border-(--border-strong)"
                />
              </li>
            );
          })}
        </ol>
        <div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-8 gap-1.5"
            onClick={addLine}
            disabled={disabled || scene.speakers.length === 0}
          >
            <Plus aria-hidden="true" />
            Add line
          </Button>
        </div>
      </section>
    </div>
  );
}
