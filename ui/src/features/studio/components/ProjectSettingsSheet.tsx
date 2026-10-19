import { useEffect, useState, type FormEvent } from "react";
import { AlertCircle, ArrowRight, Plus, X } from "lucide-react";

import type {
  ModelInfo,
  SavedVoiceSummary,
  StudioProjectPronunciationRecord,
  StudioProjectRecord,
} from "@/api";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { Slider } from "@/components/ui/slider";
import {
  StudioVoicePicker,
  type StudioModelDownloads,
} from "@/features/studio/components/StudioVoicePicker";
import {
  paragraphVoice,
  projectVoice,
  voiceOptionKey,
  type StudioVoice,
} from "@/features/studio/studio";

const sectionTitle = "text-[13px] font-semibold text-[var(--text-primary)]";

function sameVoice(left: StudioVoice, right: StudioVoice): boolean {
  return left.modelId === right.modelId && voiceOptionKey(left) === voiceOptionKey(right);
}

interface ProjectSettingsSheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  project: StudioProjectRecord;
  models: ModelInfo[];
  savedVoices: SavedVoiceSummary[];
  pronunciations: StudioProjectPronunciationRecord[];
  downloads: StudioModelDownloads;
  onSave: (voice: StudioVoice, speed: number | null) => Promise<boolean>;
  onAddRule: (source: string, replacement: string) => Promise<boolean>;
  onRemoveRule: (ruleId: string) => void;
  onError: (message: string) => void;
}

/** The project's voice, speed and pronunciation rules. */
export function ProjectSettingsSheet({
  open,
  onOpenChange,
  project,
  models,
  savedVoices,
  pronunciations,
  downloads,
  onSave,
  onAddRule,
  onRemoveRule,
  onError,
}: ProjectSettingsSheetProps) {
  const [voice, setVoice] = useState<StudioVoice>(projectVoice(project));
  const [speed, setSpeed] = useState(project.speed ?? 1);
  const [saving, setSaving] = useState(false);
  const [word, setWord] = useState("");
  const [sayAs, setSayAs] = useState("");

  useEffect(() => {
    if (!open) return;
    setVoice(projectVoice(project));
    setSpeed(project.speed ?? 1);
    // Reset the draft only when the sheet opens.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const model = models.find((candidate) => candidate.variant === voice.modelId);
  const supportsSpeed = model?.speech_capabilities?.supports_speed_control ?? false;
  const voiceChanged = !sameVoice(voice, projectVoice(project));
  const speedChanged = supportsSpeed && Math.abs(speed - (project.speed ?? 1)) > 0.001;
  const withAudio = project.segments.filter((segment) => segment.speech_record_id);
  const ownVoices = project.segments.filter((segment) => paragraphVoice(project, segment)).length;
  const affected = speedChanged
    ? withAudio.length
    : voiceChanged
      ? withAudio.filter((segment) => !paragraphVoice(project, segment)).length
      : 0;

  const save = async () => {
    setSaving(true);
    const ok = await onSave(voice, supportsSpeed ? speed : null);
    setSaving(false);
    if (ok) onOpenChange(false);
  };

  const addRule = async (event: FormEvent) => {
    event.preventDefault();
    if (!word.trim() || !sayAs.trim()) return;
    if (await onAddRule(word.trim(), sayAs.trim())) {
      setWord("");
      setSayAs("");
    }
  };

  const input =
    "h-8 min-w-0 rounded-[8px] border border-[var(--border-subtle)] bg-[var(--surface-panel-alt)] px-2 text-[13px] text-[var(--text-primary)] outline-none placeholder:text-[var(--text-subtle)] focus:border-[var(--border-strong)]";

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        aria-describedby={undefined}
        className="flex w-full max-w-[420px] flex-col gap-0 border-l border-[var(--border-default)] bg-[var(--bg-surface-1)] p-0 sm:max-w-[420px] [&>button:last-child]:hidden"
      >
        <div className="flex h-16 shrink-0 items-center justify-between border-b border-[var(--border-muted)] pl-6 pr-3">
          <SheetTitle className="text-base font-semibold text-[var(--text-primary)]">
            Project settings
          </SheetTitle>
          <button
            type="button"
            onClick={() => onOpenChange(false)}
            aria-label="Close"
            className="flex h-10 w-10 items-center justify-center rounded-[10px] text-[var(--text-muted)] transition-colors hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)]"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex min-h-0 flex-1 flex-col gap-7 overflow-y-auto px-6 py-6">
          <section className="flex flex-col gap-3">
            <h3 className={sectionTitle}>Voice</h3>
            <StudioVoicePicker
              models={models}
              savedVoices={savedVoices}
              value={voice}
              onChange={setVoice}
              onError={onError}
              downloads={downloads}
              disabled={saving}
            />
          </section>

          {supportsSpeed ? (
            <section className="flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <h3 id="studio-speed" className={sectionTitle}>
                  Speed
                </h3>
                <span className="text-xs tabular-nums text-[var(--text-muted)]">
                  {speed.toFixed(2)}×
                </span>
              </div>
              <Slider
                aria-labelledby="studio-speed"
                min={0.5}
                max={1.5}
                step={0.05}
                value={[speed]}
                onValueChange={([next]) => setSpeed(next ?? 1)}
                disabled={saving}
              />
            </section>
          ) : null}

          <section className="flex flex-col gap-3">
            <div className="flex flex-col gap-1">
              <h3 className={sectionTitle}>Pronunciation</h3>
              <p className="text-xs text-[var(--text-muted)]">
                Whole words, in any letter case, in every paragraph.
              </p>
            </div>
            <div className="flex flex-col overflow-hidden rounded-[12px] border border-[var(--border-subtle)]">
              {pronunciations.map((rule) => (
                <div
                  key={rule.id}
                  className="grid grid-cols-[minmax(0,1fr)_14px_minmax(0,1fr)_28px] items-center gap-2.5 border-b border-[var(--border-subtle)] py-1.5 pl-3 pr-1.5 text-[13px]"
                >
                  <span className="truncate text-[var(--text-primary)]">{rule.source_text}</span>
                  <ArrowRight aria-hidden="true" className="h-3.5 w-3.5 text-[var(--text-subtle)]" />
                  <span className="truncate text-[var(--text-muted)]">{rule.replacement_text}</span>
                  <button
                    type="button"
                    aria-label={`Remove ${rule.source_text}`}
                    onClick={() => onRemoveRule(rule.id)}
                    className="flex h-7 w-7 items-center justify-center rounded-[8px] text-[var(--text-subtle)] transition-colors hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)]"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
              <form
                onSubmit={(event) => void addRule(event)}
                className="grid grid-cols-[minmax(0,1fr)_14px_minmax(0,1fr)_28px] items-center gap-2.5 py-1.5 pl-1.5 pr-1.5"
              >
                <input
                  aria-label="Word"
                  placeholder="Word"
                  value={word}
                  onChange={(event) => setWord(event.target.value)}
                  className={input}
                />
                <ArrowRight aria-hidden="true" className="h-3.5 w-3.5 text-[var(--text-subtle)]" />
                <input
                  aria-label="Say it as"
                  placeholder="Say it as"
                  value={sayAs}
                  onChange={(event) => setSayAs(event.target.value)}
                  className={input}
                />
                <button
                  type="submit"
                  aria-label="Add rule"
                  disabled={!word.trim() || !sayAs.trim()}
                  className="flex h-7 w-7 items-center justify-center rounded-[8px] border border-[var(--border-default)] text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-surface-2)] disabled:opacity-40"
                >
                  <Plus className="h-3.5 w-3.5" />
                </button>
              </form>
            </div>
          </section>
        </div>

        <div className="flex shrink-0 flex-col gap-3 border-t border-[var(--border-muted)] px-6 py-4">
          {affected > 0 ? (
            <p className="flex items-start gap-2 text-xs text-[var(--status-warning-text)]">
              <AlertCircle aria-hidden="true" className="mt-px h-3.5 w-3.5 shrink-0" />
              {`Saving marks ${affected} ${affected === 1 ? "paragraph" : "paragraphs"} to render again.`}
              {voiceChanged && !speedChanged && ownVoices > 0
                ? ` ${ownVoices} with their own voice stay as they are.`
                : ""}
            </p>
          ) : null}
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => onOpenChange(false)}
              className="h-9 rounded-full border border-[var(--border-default)] px-4 text-[13px] text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-surface-2)]"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => void save()}
              disabled={saving || (!voiceChanged && !speedChanged)}
              className="h-9 rounded-full bg-[var(--accent-solid)] px-4 text-[13px] font-medium text-[var(--text-on-accent)] transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              {saving ? "Saving…" : "Save"}
            </button>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}
