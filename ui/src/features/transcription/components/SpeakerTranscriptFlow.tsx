import { formatClockTime } from "@/components/audio/AudioPlayer";
import type { SpeakerTranscriptBlock } from "@/features/transcription/transcript";
import { speakerAccent } from "@/utils/speakerAccent";

interface SpeakerTranscriptFlowProps {
  blocks: SpeakerTranscriptBlock[];
  activeIndex: number;
  onSeek: (seconds: number) => void;
  registerBlock: (index: number, element: HTMLElement | null) => void;
}

/**
 * A speaker-attributed transcript read as one document: a single surface,
 * each turn a paragraph under its speaker's name. Timed turns seek on click
 * and highlight while playing; untimed turns (Granite's native speaker turns)
 * are plain text.
 */
export function SpeakerTranscriptFlow({
  blocks,
  activeIndex,
  onSeek,
  registerBlock,
}: SpeakerTranscriptFlowProps) {
  return (
    <section
      aria-labelledby="speaker-transcript-heading"
      className="rounded-lg border border-(--border-muted) bg-(--bg-surface-0) px-3.5 py-3.5 sm:px-5 sm:py-4"
    >
      <h3
        id="speaker-transcript-heading"
        className="text-[13px] font-semibold tracking-wide text-(--text-primary)"
      >
        Transcript
      </h3>
      <div className="mt-3 space-y-3">
        {blocks.map((block, index) => {
          const accent = speakerAccent(block.speakerIndex);
          const active = index === activeIndex;
          const content = (
            <>
              <div className="flex items-center gap-2">
                <span
                  aria-hidden="true"
                  className="h-2 w-2 shrink-0 rounded-full"
                  style={{ backgroundColor: accent.solid }}
                />
                <span className="text-[13px] font-semibold text-(--text-primary)">
                  {block.speaker}
                </span>
                {block.timed ? (
                  <span className="text-[11px] font-medium text-(--text-muted)">
                    {formatClockTime(block.start)}
                  </span>
                ) : null}
              </div>
              <p className="mt-0.5 text-[15px] leading-7 text-(--text-secondary)">
                {block.text}
              </p>
            </>
          );
          const ref = (element: HTMLElement | null) => registerBlock(index, element);

          if (!block.timed) {
            return (
              <div key={`turn-${index}`} ref={ref} data-active="false">
                {content}
              </div>
            );
          }

          return (
            <button
              key={`turn-${block.start}-${index}`}
              type="button"
              ref={ref}
              onClick={() => onSeek(block.start)}
              data-active={active ? "true" : "false"}
              className="-mx-2 block w-[calc(100%+1rem)] rounded-md px-2 py-1 text-left transition-colors"
              style={{ backgroundColor: active ? accent.soft : undefined }}
            >
              {content}
            </button>
          );
        })}
      </div>
    </section>
  );
}
