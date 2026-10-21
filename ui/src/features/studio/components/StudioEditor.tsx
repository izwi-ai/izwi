import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent } from "react";
import {
  ChevronDown,
  Download,
  Loader2,
  MoreHorizontal,
  Pencil,
  Plus,
  Settings2,
  Square,
  Trash2,
  X,
} from "lucide-react";

import { api, type ModelInfo, type SavedVoiceSummary, type StudioProjectRecord } from "@/api";
import { ConfirmDeleteDialog } from "@/components/history/ConfirmDeleteDialog";
import { formatClockDuration } from "@/components/history/format";
import { PageHeader } from "@/components/PageShell";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ProjectSettingsSheet } from "@/features/studio/components/ProjectSettingsSheet";
import { ScriptParagraph, type FocusRequest } from "@/features/studio/components/ScriptParagraph";
import { StudioExportDialog } from "@/features/studio/components/StudioExportDialog";
import { StudioPlayerBar } from "@/features/studio/components/StudioPlayerBar";
import {
  ModelDownloadNotice,
  type StudioModelDownloads,
} from "@/features/studio/components/StudioVoicePicker";
import { useProjectPlayer } from "@/features/studio/hooks/useProjectPlayer";
import type { StudioProjectState } from "@/features/studio/hooks/useStudioProject";
import {
  isModelOnDevice,
  mainAction,
  modelName,
  paragraphVoice,
  projectDurationSecs,
  projectVoice,
  voiceName,
  voiceOptions,
  type StudioVoice,
  type StudioVoiceOption,
} from "@/features/studio/studio";
import { cn } from "@/lib/utils";

const pillButton =
  "inline-flex h-9 items-center gap-2 rounded-full px-3.5 text-[13px] font-medium transition-colors disabled:opacity-50";
const outlineButton = `${pillButton} border border-(--border-default) text-(--text-primary) hover:bg-(--bg-surface-2)`;
const primaryButton = `${pillButton} bg-(--accent-solid) text-(--text-on-accent) hover:opacity-90`;

interface StudioEditorProps {
  state: StudioProjectState & { project: StudioProjectRecord };
  models: ModelInfo[];
  savedVoices: SavedVoiceSummary[];
  downloads: StudioModelDownloads;
  onError: (message: string) => void;
  onDeleted: () => void;
}

type DeleteTarget = { kind: "paragraphs"; ids: string[] } | { kind: "project" };

export function StudioEditor({
  state,
  models,
  savedVoices,
  downloads,
  onError,
  onDeleted,
}: StudioEditorProps) {
  const { project, pronunciations, mutate, render, cancelRender } = state;
  const player = useProjectPlayer(project);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [exportIds, setExportIds] = useState<string[] | null | undefined>(undefined);
  const [deleteTarget, setDeleteTarget] = useState<DeleteTarget | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const selectionAnchor = useRef<string | null>(null);
  const [composer, setComposer] = useState<{ afterId: string | null; text: string } | null>(null);
  const [focusRequest, setFocusRequest] = useState<FocusRequest | null>(null);
  const [drag, setDrag] = useState<{ id: string; overId: string | null; after: boolean } | null>(null);
  const pendingSaves = useRef(new Set<Promise<unknown>>());
  const [savingCount, setSavingCount] = useState(0);

  const segmentIds = useMemo(() => project.segments.map((segment) => segment.id), [project.segments]);
  // Drop selections of paragraphs that no longer exist.
  useEffect(() => {
    setSelected((current) => current.filter((id) => segmentIds.includes(id)));
  }, [segmentIds]);

  const model = models.find((candidate) => candidate.variant === project.model_id);
  const currentVoice = projectVoice(project);
  const projectVoiceName = voiceName(currentVoice, savedVoices);
  const choices = useMemo(() => voiceOptions(model, savedVoices), [model, savedVoices]);
  const action = mainAction(project);

  /** Wait for paragraph saves in flight before changing structure. */
  const settle = useCallback(async () => {
    await Promise.all([...pendingSaves.current]);
  }, []);

  const track = useCallback(<T,>(work: Promise<T>): Promise<T> => {
    pendingSaves.current.add(work);
    setSavingCount(pendingSaves.current.size);
    void work.finally(() => {
      pendingSaves.current.delete(work);
      setSavingCount(pendingSaves.current.size);
    });
    return work;
  }, []);

  const saveText = (segmentId: string, text: string) =>
    track(
      mutate(
        () => api.updateStudioProjectSegment(project.id, segmentId, { text }),
        "Couldn't save the paragraph.",
      ).then(Boolean),
    );

  const structural = async (
    change: () => Promise<StudioProjectRecord>,
    fallback: string,
  ) => {
    await settle();
    return mutate(change, fallback);
  };

  const renderParagraphs = async (ids?: string[]) => {
    await settle();
    return render(ids);
  };

  const split = async (index: number, before: string, after: string) => {
    const segment = project.segments[index];
    const next = await structural(
      () =>
        api.splitStudioProjectSegment(project.id, segment.id, {
          before_text: before,
          after_text: after,
        }),
      "Couldn't split the paragraph.",
    );
    const created = next?.segments[index + 1];
    if (created) setFocusRequest({ segmentId: created.id, caret: 0 });
  };

  const mergeInto = async (index: number) => {
    const target = project.segments[index];
    const caret = target.text.length + 2;
    const next = await structural(
      () => api.mergeStudioProjectSegmentWithNext(project.id, target.id),
      "Couldn't merge the paragraphs.",
    );
    if (next) setFocusRequest({ segmentId: target.id, caret });
  };

  const reorder = (ids: string[]) =>
    structural(
      () => api.reorderStudioProjectSegments(project.id, { ordered_segment_ids: ids }),
      "Couldn't move the paragraph.",
    );

  const move = (index: number, direction: -1 | 1) => {
    const ids = [...segmentIds];
    const target = index + direction;
    if (target < 0 || target >= ids.length) return;
    [ids[index], ids[target]] = [ids[target], ids[index]];
    void reorder(ids);
  };

  const setParagraphVoice = (segmentId: string, choice: StudioVoiceOption | null) =>
    void mutate(
      () =>
        api.updateStudioProjectSegment(
          project.id,
          segmentId,
          choice
            ? {
                voice_mode: choice.voiceMode,
                speaker: choice.speaker ?? undefined,
                saved_voice_id: choice.savedVoiceId ?? undefined,
              }
            : { use_project_voice: true },
        ),
      "Couldn't change the paragraph's voice.",
    );

  const toggleSelect = (segmentId: string, range: boolean) => {
    setSelected((current) => {
      if (range && selectionAnchor.current) {
        const from = segmentIds.indexOf(selectionAnchor.current);
        const to = segmentIds.indexOf(segmentId);
        const span = segmentIds.slice(Math.min(from, to), Math.max(from, to) + 1);
        return Array.from(new Set([...current, ...span]));
      }
      selectionAnchor.current = segmentId;
      return current.includes(segmentId)
        ? current.filter((id) => id !== segmentId)
        : [...current, segmentId];
    });
  };

  const confirmDelete = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    if (deleteTarget.kind === "project") {
      try {
        await api.deleteStudioProject(project.id);
        onDeleted();
      } catch (err) {
        onError(err instanceof Error ? err.message : "Couldn't delete the project.");
      }
    } else {
      const ids = deleteTarget.ids;
      await structural(
        () =>
          ids.length === 1
            ? api.deleteStudioProjectSegment(project.id, ids[0])
            : api.bulkDeleteStudioProjectSegments(project.id, { segment_ids: ids }),
        "Couldn't delete the paragraphs.",
      );
      setSelected([]);
    }
    setDeleting(false);
    setDeleteTarget(null);
  };

  const addParagraph = async () => {
    if (!composer) return;
    const text = composer.text.trim();
    const afterId = composer.afterId;
    setComposer(null);
    if (!text) return;
    const next = await structural(
      () =>
        api.createStudioProjectSegment(project.id, {
          text,
          after_segment_id: afterId ?? undefined,
        }),
      "Couldn't add the paragraph.",
    );
    const position = afterId ? (next?.segments.findIndex((segment) => segment.id === afterId) ?? -2) + 1 : -1;
    const created = position >= 0 ? next?.segments[position] : next?.segments[next.segments.length - 1];
    if (created) setFocusRequest({ segmentId: created.id, caret: created.text.length });
  };

  const saveSettings = async (voice: StudioVoice, speed: number | null) =>
    Boolean(
      await mutate(
        () =>
          api.updateStudioProject(project.id, {
            model_id: voice.modelId,
            voice_mode: voice.voiceMode,
            speaker: voice.speaker ?? undefined,
            saved_voice_id: voice.savedVoiceId ?? undefined,
            speed: speed ?? undefined,
          }),
        "Couldn't save the project settings.",
      ),
    );

  const onDragOver = (segmentId: string) => (event: DragEvent) => {
    if (!drag) return;
    event.preventDefault();
    const box = event.currentTarget.getBoundingClientRect();
    const after = event.clientY > box.top + box.height / 2;
    if (drag.overId !== segmentId || drag.after !== after) {
      setDrag({ ...drag, overId: segmentId, after });
    }
  };

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    if (!drag?.overId || drag.overId === drag.id) {
      setDrag(null);
      return;
    }
    const ids = segmentIds.filter((id) => id !== drag.id);
    const at = ids.indexOf(drag.overId) + (drag.after ? 1 : 0);
    ids.splice(at, 0, drag.id);
    setDrag(null);
    void reorder(ids);
  };

  const duration = projectDurationSecs(project);
  const meta = [
    `${project.segments.length} ${project.segments.length === 1 ? "paragraph" : "paragraphs"}`,
    duration > 0 ? formatClockDuration(duration) : null,
    savingCount > 0 ? "Saving…" : "Saved",
  ]
    .filter(Boolean)
    .join(" · ");

  const currentIndex = player.segmentId ? segmentIds.indexOf(player.segmentId) : -1;

  return (
    <div className="flex min-h-full flex-col">
      <PageHeader
        className="mb-6 sm:flex-wrap"
        titleClassName="min-w-48"
        breadcrumbs={[{ label: "Studio", to: "/studio" }, { label: project.name }]}
        title={
          renaming ? (
            <input
              autoFocus
              aria-label="Project name"
              defaultValue={project.name}
              onBlur={(event) => {
                const name = event.target.value.trim();
                setRenaming(false);
                if (name && name !== project.name) {
                  void mutate(
                    () => api.updateStudioProject(project.id, { name }),
                    "Couldn't rename the project.",
                  );
                }
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter") event.currentTarget.blur();
                if (event.key === "Escape") setRenaming(false);
              }}
              className="w-full rounded-[8px] bg-transparent text-[length:inherit] font-[inherit] text-(--text-primary) outline-hidden ring-1 ring-(--border-strong)"
            />
          ) : (
            <button
              type="button"
              onClick={() => setRenaming(true)}
              title="Rename"
              className="max-w-full truncate text-left"
            >
              {project.name}
            </button>
          )
        }
        description={meta}
        actions={
          <div className="flex flex-wrap items-center justify-end gap-2">
            <button
              type="button"
              onClick={() => setSettingsOpen(true)}
              className="inline-flex h-9 items-center gap-2 rounded-full border border-(--border-default) pl-1.5 pr-3 text-[13px] text-(--text-primary) transition-colors hover:bg-(--bg-surface-2)"
            >
              <span className="flex h-6 w-6 items-center justify-center rounded-full border border-(--border-default) bg-(--surface-panel-alt) text-[11px] font-semibold">
                {projectVoiceName.charAt(0).toUpperCase()}
              </span>
              {projectVoiceName}
              <span className="hidden text-(--text-subtle) lg:inline">
                · {modelName(project.model_id, models)}
              </span>
              <ChevronDown className="h-3.5 w-3.5 text-(--text-subtle)" />
            </button>
            {action.kind !== "export" ? (
              <button type="button" onClick={() => setExportIds(null)} className={outlineButton}>
                <Download className="h-4 w-4" />
                Export
              </button>
            ) : null}
            {action.kind === "rendering" ? (
              <span className="inline-flex h-9 items-center gap-2.5 rounded-full border border-(--border-subtle) bg-(--surface-panel) pl-3.5 pr-1.5 text-[13px] text-(--text-primary)">
                <Loader2 aria-hidden="true" className="h-3.5 w-3.5 animate-spin" />
                Rendering {action.current} of {action.total}
                <button
                  type="button"
                  aria-label="Stop rendering"
                  onClick={() => void cancelRender()}
                  className="flex h-6 w-6 items-center justify-center rounded-full bg-(--surface-raised) transition-colors hover:bg-(--surface-strong)"
                >
                  <Square className="h-2.5 w-2.5 fill-current" />
                </button>
              </span>
            ) : action.kind === "render" ? (
              <button type="button" onClick={() => void renderParagraphs()} className={primaryButton}>
                {action.label}
              </button>
            ) : (
              <button type="button" onClick={() => setExportIds(null)} className={primaryButton}>
                <Download className="h-4 w-4" />
                Export
              </button>
            )}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  type="button"
                  aria-label="More project actions"
                  className="flex h-9 w-9 items-center justify-center rounded-full border border-(--border-default) text-(--text-muted) transition-colors hover:bg-(--bg-surface-2) hover:text-(--text-primary)"
                >
                  <MoreHorizontal className="h-4 w-4" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuItem onSelect={() => setRenaming(true)}>
                  <Pencil className="mr-2 h-4 w-4" />
                  Rename
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={() => setSettingsOpen(true)}>
                  <Settings2 className="mr-2 h-4 w-4" />
                  Project settings
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onSelect={() => setDeleteTarget({ kind: "project" })}
                  className="text-(--danger-text) focus:text-(--danger-text)"
                >
                  <Trash2 className="mr-2 h-4 w-4" />
                  Delete project
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        }
      />

      <div className="mx-auto flex w-full max-w-[760px] flex-1 flex-col gap-1">
        {model && !isModelOnDevice(model) ? (
          <div className="mb-3">
            <ModelDownloadNotice model={model} downloads={downloads} />
          </div>
        ) : null}

        {project.segments.map((segment, index) => {
          const own = paragraphVoice(project, segment);
          return (
            <div key={segment.id}>
              <ScriptParagraph
                segment={segment}
                number={index + 1}
                isFirst={index === 0}
                isLast={index === project.segments.length - 1}
                canDelete={project.segments.length > 1}
                playback={{
                  current: player.segmentId === segment.id,
                  playing: player.playing,
                  time: player.time,
                }}
                selected={selected.includes(segment.id)}
                ownVoiceName={own ? voiceName(own, savedVoices) : null}
                projectVoiceName={projectVoiceName}
                voiceChoices={choices}
                focusRequest={focusRequest}
                onFocusHandled={() => setFocusRequest(null)}
                dropIndicator={
                  drag && drag.overId === segment.id && drag.id !== segment.id
                    ? drag.after
                      ? "after"
                      : "before"
                    : null
                }
                onSave={(text) => saveText(segment.id, text)}
                onSplit={(before, after) => void split(index, before, after)}
                onMergeIntoPrevious={() => void mergeInto(index - 1)}
                onMergeNext={() => void mergeInto(index)}
                onMove={(direction) => move(index, direction)}
                onDelete={() => setDeleteTarget({ kind: "paragraphs", ids: [segment.id] })}
                onRender={() => void renderParagraphs([segment.id])}
                onPlay={() => player.toggleParagraph(segment.id)}
                onPlayFrom={() => player.playFrom(segment.id)}
                onToggleSelect={(range) => toggleSelect(segment.id, range)}
                onVoice={(choice) => setParagraphVoice(segment.id, choice)}
                onDragStart={(event) => {
                  event.dataTransfer.effectAllowed = "move";
                  setDrag({ id: segment.id, overId: null, after: false });
                }}
                onDragOver={onDragOver(segment.id)}
                onDrop={onDrop}
                onDragEnd={() => setDrag(null)}
              />
              {composer?.afterId === segment.id ? (
                <ParagraphComposer
                  text={composer.text}
                  onChange={(text) => setComposer({ afterId: segment.id, text })}
                  onDone={() => void addParagraph()}
                  onCancel={() => setComposer(null)}
                />
              ) : index < project.segments.length - 1 ? (
                <InsertDivider
                  label={`Add paragraph after paragraph ${index + 1}`}
                  onClick={() => setComposer({ afterId: segment.id, text: "" })}
                />
              ) : null}
            </div>
          );
        })}

        {composer?.afterId === null ? (
          <ParagraphComposer
            text={composer.text}
            onChange={(text) => setComposer({ afterId: null, text })}
            onDone={() => void addParagraph()}
            onCancel={() => setComposer(null)}
          />
        ) : (
          <button
            type="button"
            onClick={() => setComposer({ afterId: null, text: "" })}
            className="mt-2 inline-flex h-9 items-center gap-2 self-start rounded-full px-3 text-[13px] text-(--text-muted) transition-colors hover:bg-(--bg-surface-2) hover:text-(--text-primary)"
          >
            <Plus className="h-4 w-4" />
            Add paragraph
          </button>
        )}
      </div>

      {selected.length > 0 ? (
        <div className="sticky bottom-[76px] z-20 mx-auto mt-4 flex items-center gap-1 rounded-full border border-(--border-default) bg-(--bg-surface-1) py-1.5 pl-4 pr-1.5 text-[13px] shadow-(--shadow-raised)">
          <span className="mr-2 text-(--text-primary)">{selected.length} selected</span>
          <button type="button" onClick={() => void renderParagraphs(selected)} className="h-8 rounded-full px-3 text-(--text-primary) hover:bg-(--bg-surface-2)">
            Render
          </button>
          <button type="button" onClick={() => setExportIds(selected)} className="h-8 rounded-full px-3 text-(--text-primary) hover:bg-(--bg-surface-2)">
            Export
          </button>
          <button
            type="button"
            disabled={selected.length >= project.segments.length}
            onClick={() => setDeleteTarget({ kind: "paragraphs", ids: selected })}
            className="h-8 rounded-full px-3 text-(--danger-text) hover:bg-(--danger-bg) disabled:opacity-40"
          >
            Delete
          </button>
          <button
            type="button"
            aria-label="Clear selection"
            onClick={() => setSelected([])}
            className="flex h-8 w-8 items-center justify-center rounded-full bg-(--surface-raised) text-(--text-muted) hover:text-(--text-primary)"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      ) : null}

      <StudioPlayerBar
        player={player}
        currentNumber={currentIndex >= 0 ? currentIndex + 1 : null}
        paragraphCount={project.segments.length}
      />

      <ProjectSettingsSheet
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        project={project}
        models={models}
        savedVoices={savedVoices}
        pronunciations={pronunciations}
        downloads={downloads}
        onSave={saveSettings}
        onAddRule={state.addPronunciation}
        onRemoveRule={(ruleId) => void state.removePronunciation(ruleId)}
        onError={onError}
      />
      <StudioExportDialog
        open={exportIds !== undefined}
        onClose={() => setExportIds(undefined)}
        project={project}
        segmentIds={exportIds ?? undefined}
        onRender={renderParagraphs}
      />
      <ConfirmDeleteDialog
        noun={
          deleteTarget?.kind === "project"
            ? "project"
            : deleteTarget
              ? deleteTarget.ids.length === 1
                ? "paragraph"
                : `${deleteTarget.ids.length} paragraphs`
              : null
        }
        itemLabel={deleteTarget?.kind === "project" ? project.name : undefined}
        description={
          deleteTarget?.kind === "project"
            ? "Its script and audio are deleted. This can't be undone."
            : "Their text and audio are deleted. This can't be undone."
        }
        pending={deleting}
        error={null}
        onConfirm={() => void confirmDelete()}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  );
}

function InsertDivider({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <div className="group/insert flex h-3 items-center gap-2.5 pl-[54px] pr-3">
      <span className="h-px flex-1 bg-(--border-subtle) opacity-0 transition-opacity group-hover/insert:opacity-100" />
      <button
        type="button"
        aria-label={label}
        onClick={onClick}
        className="flex h-5 w-5 items-center justify-center rounded-full border border-(--border-default) bg-(--surface-canvas) text-(--text-muted) opacity-0 transition-opacity hover:text-(--text-primary) focus-visible:opacity-100 group-hover/insert:opacity-100"
      >
        <Plus className="h-3 w-3" />
      </button>
      <span className="h-px flex-1 bg-(--border-subtle) opacity-0 transition-opacity group-hover/insert:opacity-100" />
    </div>
  );
}

function ParagraphComposer({
  text,
  onChange,
  onDone,
  onCancel,
}: {
  text: string;
  onChange: (text: string) => void;
  onDone: () => void;
  onCancel: () => void;
}) {
  // Enter, Escape and the blur that follows them finish the composer once.
  const finished = useRef(false);
  const finish = (submit: boolean) => {
    if (finished.current) return;
    finished.current = true;
    if (submit) onDone();
    else onCancel();
  };
  return (
    <div className={cn("my-1 rounded-[12px] border border-(--border-default) bg-(--surface-panel) px-3 py-3 pl-[54px]")}>
      <textarea
        autoFocus
        rows={2}
        value={text}
        aria-label="New paragraph"
        placeholder="Write the new paragraph, then press Enter"
        onChange={(event) => onChange(event.target.value)}
        onBlur={() => finish(true)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault();
            finish(true);
          }
          if (event.key === "Escape") finish(false);
        }}
        className="w-full resize-none bg-transparent text-[15px] leading-[1.65] text-(--text-primary) outline-hidden placeholder:text-(--text-subtle)"
      />
    </div>
  );
}
