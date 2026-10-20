import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type DragEvent,
  type KeyboardEvent,
  type MouseEvent,
} from "react";
import {
  AlertCircle,
  ArrowDown,
  ArrowUp,
  Check,
  GripVertical,
  Loader2,
  Merge,
  MoreHorizontal,
  Pause,
  Play,
  RotateCw,
  Scissors,
  Trash2,
  UserRound,
} from "lucide-react";

import type { StudioProjectSegmentRecord } from "@/api";
import { formatClockDuration } from "@/components/history/format";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import type { StudioVoiceOption } from "@/features/studio/studio";
import { cn } from "@/lib/utils";

const AUTOSAVE_DELAY_MS = 800;

export interface ParagraphPlayback {
  /** This paragraph is the player's current one. */
  current: boolean;
  playing: boolean;
  /** Seconds into this paragraph. */
  time: number;
}

/** Where to put the caret once a paragraph mounts or updates. */
export interface FocusRequest {
  segmentId: string;
  caret: number;
}

interface ScriptParagraphProps {
  segment: StudioProjectSegmentRecord;
  number: number;
  isFirst: boolean;
  isLast: boolean;
  canDelete: boolean;
  playback: ParagraphPlayback;
  selected: boolean;
  /** The paragraph's own voice name, or null when it follows the project. */
  ownVoiceName: string | null;
  projectVoiceName: string;
  voiceChoices: StudioVoiceOption[];
  focusRequest: FocusRequest | null;
  onFocusHandled: () => void;
  dropIndicator: "before" | "after" | null;
  /** Resolves false when the save failed; the edit then stays unsaved. */
  onSave: (text: string) => Promise<boolean>;
  onSplit: (before: string, after: string) => void;
  onMergeIntoPrevious: () => void;
  onMergeNext: () => void;
  onMove: (direction: -1 | 1) => void;
  onDelete: () => void;
  onRender: () => void;
  onPlay: () => void;
  onPlayFrom: () => void;
  onToggleSelect: (range: boolean) => void;
  onVoice: (choice: StudioVoiceOption | null) => void;
  onDragStart: (event: DragEvent) => void;
  onDragOver: (event: DragEvent) => void;
  onDrop: (event: DragEvent) => void;
  onDragEnd: () => void;
}

const iconButton =
  "flex h-[30px] w-[30px] items-center justify-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--border-strong)]";

export function ScriptParagraph({
  segment,
  number,
  isFirst,
  isLast,
  canDelete,
  playback,
  selected,
  ownVoiceName,
  projectVoiceName,
  voiceChoices,
  focusRequest,
  onFocusHandled,
  dropIndicator,
  onSave,
  onSplit,
  onMergeIntoPrevious,
  onMergeNext,
  onMove,
  onDelete,
  onRender,
  onPlay,
  onPlayFrom,
  onToggleSelect,
  onVoice,
  onDragStart,
  onDragOver,
  onDrop,
  onDragEnd,
}: ScriptParagraphProps) {
  const [draft, setDraft] = useState(segment.text);
  const [saving, setSaving] = useState(false);
  // The text the server has: a ref for handlers, state for rendering.
  const saved = useRef(segment.text);
  const [savedText, setSavedText] = useState(segment.text);
  const markSaved = (text: string) => {
    saved.current = text;
    setSavedText(text);
  };
  const caret = useRef(0);
  const textarea = useRef<HTMLTextAreaElement>(null);

  // Take the server's text unless the paragraph has edits of its own.
  useEffect(() => {
    setDraft((current) => (current === saved.current ? segment.text : current));
    saved.current = segment.text;
    setSavedText(segment.text);
  }, [segment.text]);

  const save = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || trimmed === saved.current) return;
    const previous = saved.current;
    markSaved(trimmed);
    setSaving(true);
    const ok = await onSave(trimmed);
    if (!ok && saved.current === trimmed) markSaved(previous);
    setSaving(false);
  };
  const saveRef = useRef(save);
  useEffect(() => {
    saveRef.current = save;
  });

  useEffect(() => {
    if (!draft.trim() || draft.trim() === savedText) return;
    const timer = window.setTimeout(() => void saveRef.current(draft), AUTOSAVE_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [draft, savedText]);

  useLayoutEffect(() => {
    const element = textarea.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${element.scrollHeight}px`;
  }, [draft]);

  // Wait for the textarea to show the server's text (after a split or merge)
  // before placing the caret in it.
  useEffect(() => {
    if (focusRequest?.segmentId !== segment.id || !textarea.current) return;
    if (draft !== segment.text) return;
    const at = Math.min(focusRequest.caret, draft.length);
    textarea.current.focus();
    textarea.current.setSelectionRange(at, at);
    onFocusHandled();
  }, [draft, focusRequest, onFocusHandled, segment.id, segment.text]);

  const dirty = draft.trim() !== savedText;
  const splitAt = (at: number) => {
    const before = draft.slice(0, at).trim();
    const after = draft.slice(at).trim();
    if (!before || !after) return;
    markSaved(before);
    setDraft(before);
    onSplit(before, after);
  };

  const handleKeyDown = async (event: KeyboardEvent<HTMLTextAreaElement>) => {
    const element = event.currentTarget;
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      splitAt(element.selectionStart);
      return;
    }
    if (
      event.key === "Backspace" &&
      !isFirst &&
      element.selectionStart === 0 &&
      element.selectionEnd === 0
    ) {
      event.preventDefault();
      // An emptied paragraph goes: merging would bring back the text the
      // user just deleted, which is still the saved text.
      if (!draft.trim()) {
        onDelete();
        return;
      }
      await save(draft);
      onMergeIntoPrevious();
    }
  };

  const state = segment.render_state;
  const hasAudio = Boolean(segment.speech_record_id);
  const duration = segment.audio_duration_secs;
  const progress =
    playback.current && duration ? Math.min(1, playback.time / duration) : 0;

  const gutterClick = (event: MouseEvent) => {
    if (event.shiftKey || event.metaKey || event.ctrlKey) {
      onToggleSelect(event.shiftKey);
      return;
    }
    if (state === "rendered" && !dirty) onPlay();
    else onRender();
  };

  const gutter = (() => {
    if (state === "rendering") {
      return (
        <span role="img" aria-label="Rendering" className={cn(iconButton, "text-[var(--text-primary)]")}>
          <Loader2 className="h-[18px] w-[18px] animate-spin" />
        </span>
      );
    }
    if (state === "queued") {
      return (
        <span role="img" aria-label="Queued" className={iconButton}>
          <span className="h-2.5 w-2.5 rounded-full border-[1.5px] border-dashed border-[var(--text-subtle)]" />
        </span>
      );
    }
    if (state === "rendered" && !dirty) {
      const active = playback.current && playback.playing;
      return (
        <button
          type="button"
          onClick={gutterClick}
          aria-label={active ? `Pause paragraph ${number}` : `Play paragraph ${number}`}
          className={cn(
            iconButton,
            active
              ? "bg-[var(--accent-solid)] text-[var(--text-on-accent)]"
              : "border border-[var(--border-default)] text-[var(--text-primary)] hover:bg-[var(--bg-surface-2)]",
          )}
        >
          {active ? <Pause className="h-3 w-3 fill-current" /> : <Play className="ml-px h-3 w-3 fill-current" />}
        </button>
      );
    }
    const failed = state === "failed";
    const changed = state === "stale" || dirty;
    return (
      <button
        type="button"
        onClick={gutterClick}
        aria-label={`Render paragraph ${number}`}
        className={cn(
          iconButton,
          failed
            ? "border border-[var(--danger-border)] text-[var(--danger-text)] hover:bg-[var(--danger-bg)]"
            : changed
              ? "border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] text-[var(--status-warning-text)]"
              : "border border-[var(--border-default)] text-[var(--text-muted)] hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)]",
        )}
      >
        <RotateCw className="h-[13px] w-[13px]" />
      </button>
    );
  })();

  const status = (() => {
    if (saving) return <span className="text-[var(--text-subtle)]">Saving…</span>;
    if (state === "rendering") return <span className="text-[var(--text-muted)]">Rendering…</span>;
    if (state === "queued") return <span className="text-[var(--text-subtle)]">Queued</span>;
    if (state === "failed") return <span className="text-[var(--danger-text)]">Failed</span>;
    if (state === "stale" || dirty) return <span className="text-[var(--status-warning-text)]">Changed</span>;
    if (playback.current && duration) {
      return (
        <span className="tabular-nums text-[var(--text-muted)]">
          {formatClockDuration(playback.time)} / {formatClockDuration(duration)}
        </span>
      );
    }
    return hasAudio ? (
      <span className="tabular-nums text-[var(--text-subtle)]">{formatClockDuration(duration)}</span>
    ) : null;
  })();

  return (
    <div
      onDragOver={onDragOver}
      onDrop={onDrop}
      className={cn(
        "group relative grid grid-cols-[30px_minmax(0,1fr)_auto] items-start gap-3 rounded-[12px] px-3 py-3 transition-colors",
        playback.current && "bg-[var(--accent-soft)]",
        selected && "bg-[var(--accent-strong)]",
        state === "failed" && "bg-[var(--danger-bg)]",
        "focus-within:bg-[var(--surface-panel)]",
      )}
      data-segment-id={segment.id}
    >
      {dropIndicator ? (
        <span
          aria-hidden="true"
          className={cn(
            "pointer-events-none absolute inset-x-3 h-0.5 rounded-full bg-[var(--text-primary)]",
            dropIndicator === "before" ? "-top-px" : "-bottom-px",
          )}
        />
      ) : null}
      <span
        draggable
        onDragStart={onDragStart}
        onDragEnd={onDragEnd}
        aria-hidden="true"
        title="Drag to reorder"
        className="absolute -left-6 top-[18px] hidden cursor-grab text-[var(--text-subtle)] opacity-0 transition-opacity group-hover:opacity-100 sm:block"
      >
        <GripVertical className="h-4 w-4" />
      </span>

      {gutter}

      <div className="flex min-w-0 flex-col gap-2">
        <textarea
          ref={textarea}
          value={draft}
          rows={1}
          aria-label={`Paragraph ${number}`}
          onChange={(event) => setDraft(event.target.value)}
          onSelect={(event) => {
            caret.current = event.currentTarget.selectionStart;
          }}
          onBlur={() => {
            if (!draft.trim()) setDraft(saved.current);
            else void save(draft);
          }}
          onKeyDown={(event) => void handleKeyDown(event)}
          className="w-full resize-none overflow-hidden bg-transparent pt-[3px] text-[15px] leading-[1.65] text-[var(--text-primary)] outline-none placeholder:text-[var(--text-subtle)]"
        />
        {state === "failed" && segment.render_error ? (
          <p className="flex items-start gap-1.5 text-xs text-[var(--danger-text)]">
            <AlertCircle aria-hidden="true" className="mt-px h-3.5 w-3.5 shrink-0" />
            {segment.render_error}
          </p>
        ) : null}
        {ownVoiceName ? (
          <span className="inline-flex items-center gap-1.5 self-start rounded-full border border-[var(--border-subtle)] px-2 py-0.5 text-xs text-[var(--text-muted)]">
            <UserRound aria-hidden="true" className="h-3 w-3" />
            {ownVoiceName}
          </span>
        ) : null}
        {playback.current ? (
          <div
            aria-hidden="true"
            className="h-[3px] overflow-hidden rounded-full bg-[var(--surface-raised)]"
          >
            <div className="h-full bg-[var(--text-primary)]" style={{ width: `${progress * 100}%` }} />
          </div>
        ) : null}
      </div>

      <div className="flex min-h-[30px] items-center justify-end gap-1 text-xs">
        <span className="whitespace-nowrap">{status}</span>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              aria-label={`More actions for paragraph ${number}`}
              className={cn(
                iconButton,
                "rounded-[8px] text-[var(--text-muted)] opacity-0 hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)] focus-visible:opacity-100 group-hover:opacity-100 group-focus-within:opacity-100 data-[state=open]:opacity-100",
              )}
            >
              <MoreHorizontal className="h-4 w-4" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            {hasAudio ? (
              <DropdownMenuItem onSelect={onPlayFrom}>
                <Play className="mr-2 h-4 w-4" />
                Play from here
              </DropdownMenuItem>
            ) : null}
            <DropdownMenuItem
              disabled={state === "queued" || state === "rendering"}
              onSelect={onRender}
            >
              <RotateCw className="mr-2 h-4 w-4" />
              {state === "rendered" ? "Render again" : "Render"}
            </DropdownMenuItem>
            {voiceChoices.length > 0 ? (
              <DropdownMenuSub>
                <DropdownMenuSubTrigger>
                  <UserRound className="mr-2 h-4 w-4" />
                  Voice for this paragraph
                </DropdownMenuSubTrigger>
                <DropdownMenuSubContent className="max-h-80 w-56 overflow-y-auto">
                  <DropdownMenuItem onSelect={() => onVoice(null)}>
                    <span className="flex min-w-0 flex-1 flex-col">
                      <span>Project voice</span>
                      <span className="text-xs text-[var(--text-muted)]">{projectVoiceName}</span>
                    </span>
                    {ownVoiceName === null ? <Check className="h-4 w-4" /> : null}
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  {voiceChoices.map((choice) => (
                    <DropdownMenuItem key={choice.key} onSelect={() => onVoice(choice)}>
                      <span className="flex-1 truncate">{choice.name}</span>
                      {ownVoiceName === choice.name ? <Check className="h-4 w-4" /> : null}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuSubContent>
              </DropdownMenuSub>
            ) : null}
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={() => splitAt(caret.current)}>
              <Scissors className="mr-2 h-4 w-4" />
              Split at cursor
              <DropdownMenuShortcut>Enter</DropdownMenuShortcut>
            </DropdownMenuItem>
            {!isLast ? (
              <DropdownMenuItem onSelect={onMergeNext}>
                <Merge className="mr-2 h-4 w-4" />
                Merge with next
              </DropdownMenuItem>
            ) : null}
            {!isFirst ? (
              <DropdownMenuItem onSelect={() => onMove(-1)}>
                <ArrowUp className="mr-2 h-4 w-4" />
                Move up
              </DropdownMenuItem>
            ) : null}
            {!isLast ? (
              <DropdownMenuItem onSelect={() => onMove(1)}>
                <ArrowDown className="mr-2 h-4 w-4" />
                Move down
              </DropdownMenuItem>
            ) : null}
            {canDelete ? (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onSelect={onDelete}
                  className="text-[var(--danger-text)] focus:text-[var(--danger-text)]"
                >
                  <Trash2 className="mr-2 h-4 w-4" />
                  Delete
                </DropdownMenuItem>
              </>
            ) : null}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
}
