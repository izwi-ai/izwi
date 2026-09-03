import { useEffect, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  Link2,
  Loader2,
  MoreHorizontal,
  PencilLine,
  Play,
  Plus,
  Scissors,
  Settings,
  Trash2,
  X,
} from "lucide-react";
import type { StudioProjectRecord, StudioProjectSegmentRecord } from "@/api";
import type { StudioSegmentDraftConflict } from "@/features/studio/segmentDrafts";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";

const INSERT_END_TARGET = "__end__";

interface StudioSegmentEditorProps {
  project: StudioProjectRecord;
  segmentDrafts: Record<string, string>;
  segmentDraftConflicts?: Record<string, StudioSegmentDraftConflict>;
  segmentSelections: Record<string, number | null>;
  selectedSegmentIdSet: ReadonlySet<string>;
  selectedSegmentCount: number;
  queuedSegmentIdSet: ReadonlySet<string>;
  savingSegmentId: string | null;
  renderingSegmentId: string | null;
  addingSegmentAfterSegmentId: string | null;
  focusSegmentId: string | null;
  onToggleSelectAll: () => void;
  onRenderSelected: () => void;
  onDeleteSelected: () => void;
  onAddSegment: (afterSegmentId: string | null, text: string) => Promise<boolean>;
  onToggleSegmentSelection: (segmentId: string, checked: boolean) => void;
  onSaveSegment: (segmentId: string) => void;
  onMoveSegment: (segmentId: string, direction: "up" | "down") => void;
  onMergeSegmentWithNext: (segmentId: string) => void;
  onSplitSegment: (segmentId: string) => void;
  onRenderSegment: (segmentId: string) => void;
  onDeleteSegment: (segmentId: string) => void;
  onOpenSegmentSettings: (segmentId: string) => void;
  onChangeSegmentDraft: (segmentId: string, value: string) => void;
  onRestoreSegmentDraftConflict?: (segmentId: string) => void;
  onDiscardSegmentDraftConflict?: (segmentId: string) => void;
  onChangeSegmentCursor: (segmentId: string, cursor: number | null) => void;
  onFocusSegmentHandled: (segmentId: string) => void;
  audioUrlForRecordId: (recordId: string) => string;
}

function SegmentActionsMenu({
  segment,
  isFirst,
  isLast,
  canSplitSegment,
  canDeleteSegment,
  onMoveSegment,
  onMergeSegmentWithNext,
  onSplitSegment,
  onDeleteSegment,
}: {
  segment: StudioProjectSegmentRecord;
  isFirst: boolean;
  isLast: boolean;
  canSplitSegment: boolean;
  canDeleteSegment: boolean;
  onMoveSegment: (segmentId: string, direction: "up" | "down") => void;
  onMergeSegmentWithNext: (segmentId: string) => void;
  onSplitSegment: (segmentId: string) => void;
  onDeleteSegment: (segmentId: string) => void;
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="icon"
          className="h-9 w-9 bg-[var(--bg-surface-0)]"
          aria-label={`More actions for segment ${segment.position + 1}`}
        >
          <MoreHorizontal className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-44">
        {!isFirst ? (
          <DropdownMenuItem onSelect={() => onMoveSegment(segment.id, "up")}>
            <ArrowUp className="mr-2 h-4 w-4" />
            Move up
          </DropdownMenuItem>
        ) : null}
        {!isLast ? (
          <DropdownMenuItem onSelect={() => onMoveSegment(segment.id, "down")}>
            <ArrowDown className="mr-2 h-4 w-4" />
            Move down
          </DropdownMenuItem>
        ) : null}
        {!isLast ? (
          <DropdownMenuItem onSelect={() => onMergeSegmentWithNext(segment.id)}>
            <Link2 className="mr-2 h-4 w-4" />
            Merge next
          </DropdownMenuItem>
        ) : null}
        {canSplitSegment ? (
          <DropdownMenuItem onSelect={() => onSplitSegment(segment.id)}>
            <Scissors className="mr-2 h-4 w-4" />
            Split at cursor
          </DropdownMenuItem>
        ) : null}
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onSelect={() => onDeleteSegment(segment.id)}
          disabled={!canDeleteSegment}
          className="text-[var(--danger-text)] focus:text-[var(--danger-text)]"
        >
          <Trash2 className="mr-2 h-4 w-4" />
          Delete
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function InsertSegmentControls({
  targetKey,
  label,
  isComposerOpen,
  draftText,
  errorMessage,
  isSubmitting,
  disableToggle,
  onToggleComposer,
  onChangeText,
  onSubmit,
  onCancel,
}: {
  targetKey: string;
  label: string;
  isComposerOpen: boolean;
  draftText: string;
  errorMessage: string | null;
  isSubmitting: boolean;
  disableToggle: boolean;
  onToggleComposer: (targetKey: string) => void;
  onChangeText: (value: string) => void;
  onSubmit: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3 px-1">
        <div className="h-px flex-1 bg-[var(--border-muted)]" />
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => onToggleComposer(targetKey)}
          disabled={disableToggle}
          className="h-8 rounded-full border-dashed bg-[var(--bg-surface-0)] px-3 text-xs"
        >
          <Plus className="h-3.5 w-3.5" />
          {label}
        </Button>
        <div className="h-px flex-1 bg-[var(--border-muted)]" />
      </div>

      {isComposerOpen ? (
        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-3">
          <Textarea
            value={draftText}
            onChange={(event) => onChangeText(event.target.value)}
            placeholder="Type the new segment text..."
            className="min-h-[88px] border-[var(--border-muted)] bg-[var(--bg-surface-1)] text-sm"
          />
          <div className="mt-2 flex flex-wrap items-center justify-end gap-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={onCancel}
              disabled={isSubmitting}
            >
              <X className="h-4 w-4" />
              Cancel
            </Button>
            <Button type="button" size="sm" onClick={onSubmit} disabled={isSubmitting}>
              {isSubmitting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Plus className="h-4 w-4" />
              )}
              Add segment
            </Button>
          </div>
          {errorMessage ? (
            <div className="mt-2 text-xs text-[var(--danger-text)]">{errorMessage}</div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

export function StudioSegmentEditor({
  project,
  segmentDrafts,
  segmentDraftConflicts = {},
  segmentSelections,
  selectedSegmentIdSet,
  selectedSegmentCount,
  queuedSegmentIdSet,
  savingSegmentId,
  renderingSegmentId,
  addingSegmentAfterSegmentId,
  focusSegmentId,
  onToggleSelectAll,
  onRenderSelected,
  onDeleteSelected,
  onAddSegment,
  onToggleSegmentSelection,
  onSaveSegment,
  onMoveSegment,
  onMergeSegmentWithNext,
  onSplitSegment,
  onRenderSegment,
  onDeleteSegment,
  onOpenSegmentSettings,
  onChangeSegmentDraft,
  onRestoreSegmentDraftConflict,
  onDiscardSegmentDraftConflict,
  onChangeSegmentCursor,
  onFocusSegmentHandled,
  audioUrlForRecordId,
}: StudioSegmentEditorProps) {
  const allSelected =
    project.segments.length > 0 && selectedSegmentCount === project.segments.length;
  const [activeInsertTargetKey, setActiveInsertTargetKey] = useState<string | null>(
    null,
  );
  const [newSegmentDraft, setNewSegmentDraft] = useState("");
  const [newSegmentError, setNewSegmentError] = useState<string | null>(null);
  const textareaRefs = useRef<Record<string, HTMLTextAreaElement | null>>({});
  const isAddingSegment = addingSegmentAfterSegmentId !== null;

  useEffect(() => {
    if (!activeInsertTargetKey) {
      return;
    }
    if (activeInsertTargetKey === INSERT_END_TARGET) {
      return;
    }
    const segmentStillExists = project.segments.some(
      (segment) => segment.id === activeInsertTargetKey,
    );
    if (!segmentStillExists) {
      setActiveInsertTargetKey(null);
      setNewSegmentDraft("");
      setNewSegmentError(null);
    }
  }, [activeInsertTargetKey, project.segments]);

  useEffect(() => {
    if (!focusSegmentId) {
      return;
    }
    const target = textareaRefs.current[focusSegmentId];
    if (!target) {
      return;
    }
    target.focus();
    const end = target.value.length;
    target.setSelectionRange(end, end);
    onFocusSegmentHandled(focusSegmentId);
  }, [focusSegmentId, onFocusSegmentHandled, project.segments]);

  const toggleInsertComposer = (targetKey: string) => {
    if (isAddingSegment) {
      return;
    }
    if (activeInsertTargetKey === targetKey) {
      setActiveInsertTargetKey(null);
      setNewSegmentDraft("");
      setNewSegmentError(null);
      return;
    }
    setActiveInsertTargetKey(targetKey);
    setNewSegmentDraft("");
    setNewSegmentError(null);
  };

  const handleCancelInsertComposer = () => {
    if (isAddingSegment) {
      return;
    }
    setActiveInsertTargetKey(null);
    setNewSegmentDraft("");
    setNewSegmentError(null);
  };

  const handleSubmitInsertComposer = async (targetKey: string) => {
    if (isAddingSegment) {
      return;
    }
    const text = newSegmentDraft.trim();
    if (!text) {
      setNewSegmentError("Enter text for the new segment.");
      return;
    }
    setNewSegmentError(null);
    const afterSegmentId = targetKey === INSERT_END_TARGET ? null : targetKey;
    const didAdd = await onAddSegment(afterSegmentId, text);
    if (!didAdd) {
      return;
    }
    setActiveInsertTargetKey(null);
    setNewSegmentDraft("");
    setNewSegmentError(null);
  };

  return (
    <>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={onToggleSelectAll}
          className="bg-[var(--bg-surface-1)]"
        >
          {allSelected ? "Clear selection" : "Select all"}
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={onRenderSelected}
          disabled={selectedSegmentCount === 0}
          className="bg-[var(--bg-surface-1)]"
        >
          <Play className="h-4 w-4" />
          Render selected
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={onDeleteSelected}
          disabled={selectedSegmentCount === 0}
          className="bg-[var(--bg-surface-1)]"
        >
          <Trash2 className="h-4 w-4" />
          Delete selected
        </Button>
        {selectedSegmentCount > 0 ? (
          <span className="rounded-full border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] px-2.5 py-1 text-[11px] text-[var(--status-warning-text)]">
            {selectedSegmentCount} selected
          </span>
        ) : null}
      </div>

      <div className="mt-5 space-y-4">
        {project.segments.map((segment, index) => {
          const draft = segmentDrafts[segment.id] ?? segment.text;
          const draftConflict = segmentDraftConflicts[segment.id];
          const segmentDirty = draft !== segment.text;
          const segmentNeedsRender = segmentDirty || !segment.speech_record_id;
          const isSaving = savingSegmentId === segment.id;
          const isRendering = renderingSegmentId === segment.id;
          const segmentQueued = queuedSegmentIdSet.has(segment.id);
          const splitIndex = segmentSelections[segment.id];
          const canSplitSegment =
            typeof splitIndex === "number" && splitIndex > 0 && splitIndex < draft.length;
          const isSelected = selectedSegmentIdSet.has(segment.id);
          const isFirst = segment.position === 0;
          const isLast = segment.position === project.segments.length - 1;
          const canDeleteSegment = project.segments.length > 1;
          const renderButtonLabel = segmentQueued
            ? "Queued"
            : segmentDirty
              ? "Save draft first"
              : segmentNeedsRender
                ? "Render block"
                : "Re-render";

          return (
            <div key={segment.id} className="space-y-4">
              <article className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 sm:p-5">
                <div className="space-y-3">
                  <div className="min-w-0 space-y-1.5">
                    <div
                      data-testid={`studio-segment-metadata-${segment.id}`}
                      className="flex flex-wrap items-center gap-2"
                    >
                      <div className="inline-flex items-center">
                        <Switch
                          checked={isSelected}
                          onCheckedChange={(checked) =>
                            onToggleSegmentSelection(segment.id, checked)
                          }
                          aria-label={`Select segment ${segment.position + 1}`}
                          className="h-5 w-9"
                        />
                      </div>
                      <span className="whitespace-nowrap text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                        Segment {segment.position + 1}
                      </span>
                      <span
                        className={cn(
                          "whitespace-nowrap rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em]",
                          segmentDirty
                            ? "border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] text-[var(--status-warning-text)]"
                            : segment.speech_record_id
                              ? "border-[var(--status-positive-border)] bg-[var(--status-positive-bg)] text-[var(--status-positive-text)]"
                              : "border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-[var(--text-muted)]",
                        )}
                      >
                        {segmentDirty
                          ? "Edited"
                          : segment.speech_record_id
                            ? "Rendered"
                            : "Needs render"}
                      </span>
                    </div>
                    <div className="text-sm text-[var(--text-secondary)]">
                      {segment.input_chars} chars
                      {segment.audio_duration_secs
                        ? ` · ${segment.audio_duration_secs.toFixed(1)}s audio`
                        : ""}
                    </div>
                  </div>

                  <div
                    data-testid={`studio-segment-actions-${segment.id}`}
                    className="flex w-full flex-wrap items-center gap-2"
                  >
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => onSaveSegment(segment.id)}
                      disabled={!segmentDirty || isSaving}
                      className="min-w-[8rem] flex-1 justify-center bg-[var(--bg-surface-0)]"
                    >
                      {isSaving ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <PencilLine className="h-4 w-4" />
                      )}
                      Save draft
                    </Button>
                    <Button
                      size="sm"
                      onClick={() => onRenderSegment(segment.id)}
                      disabled={segmentDirty || isRendering || segmentQueued}
                      className="min-w-[8rem] flex-1 justify-center"
                    >
                      {isRendering ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Play className="h-4 w-4" />
                      )}
                      {renderButtonLabel}
                    </Button>
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => onOpenSegmentSettings(segment.id)}
                      className="ml-auto h-9 w-9 bg-[var(--bg-surface-0)]"
                      aria-label={`Open settings for segment ${segment.position + 1}`}
                    >
                      <Settings className="h-4 w-4" />
                    </Button>
                    <div>
                      <SegmentActionsMenu
                        segment={segment}
                        isFirst={isFirst}
                        isLast={isLast}
                        canSplitSegment={canSplitSegment}
                        canDeleteSegment={canDeleteSegment}
                        onMoveSegment={onMoveSegment}
                        onMergeSegmentWithNext={onMergeSegmentWithNext}
                        onSplitSegment={onSplitSegment}
                        onDeleteSegment={onDeleteSegment}
                      />
                    </div>
                  </div>
                </div>

                {draftConflict ? (
                  <div
                    className="mt-4 rounded-xl border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] px-3 py-3 text-xs text-[var(--status-warning-text)]"
                    role="alert"
                  >
                    <p className="font-semibold">Saved draft needs review</p>
                    <p className="mt-1 leading-5">
                      This segment changed after your local draft was saved, so the
                      newer server text is shown. Restore your local draft only if
                      you still want it.
                    </p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {onRestoreSegmentDraftConflict ? (
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          className="h-7 bg-[var(--bg-surface-0)]"
                          onClick={() => onRestoreSegmentDraftConflict(segment.id)}
                        >
                          Restore local draft
                        </Button>
                      ) : null}
                      {onDiscardSegmentDraftConflict ? (
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="h-7"
                          onClick={() => onDiscardSegmentDraftConflict(segment.id)}
                        >
                          Keep server text
                        </Button>
                      ) : null}
                    </div>
                  </div>
                ) : null}

                <Textarea
                  className="mt-4 border-[var(--border-muted)] bg-[var(--bg-surface-0)]"
                  value={draft}
                  disabled={Boolean(draftConflict)}
                  ref={(node) => {
                    textareaRefs.current[segment.id] = node;
                  }}
                  onChange={(event) => onChangeSegmentDraft(segment.id, event.target.value)}
                  onSelect={(event) =>
                    onChangeSegmentCursor(segment.id, event.currentTarget.selectionStart)
                  }
                />

                <div className="mt-2 text-xs text-[var(--text-muted)]">
                  {canSplitSegment
                    ? "Split at cursor is ready in the actions menu."
                    : "Place the text cursor inside this block to enable split."}
                </div>

                {segmentDirty ? (
                  <div className="mt-4 rounded-xl border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] px-3 py-2 text-xs text-[var(--status-warning-text)]">
                    This block has local edits. Save the draft explicitly before
                    rendering new audio.
                  </div>
                ) : null}

                {segment.speech_record_id ? (
                  <div className="mt-4 space-y-2 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-3">
                    <audio
                      src={audioUrlForRecordId(segment.speech_record_id)}
                      controls
                      preload="none"
                      className="h-10 w-full"
                    />
                    <div className="text-xs text-[var(--text-muted)]">
                      {segmentDirty
                        ? "Preview reflects the last rendered audio until you render this edited block again."
                        : `Linked generation: ${segment.speech_record_id}`}
                    </div>
                  </div>
                ) : null}
              </article>

              {index < project.segments.length - 1 ? (
                <InsertSegmentControls
                  targetKey={segment.id}
                  label="Add segment here"
                  isComposerOpen={activeInsertTargetKey === segment.id}
                  draftText={newSegmentDraft}
                  errorMessage={newSegmentError}
                  isSubmitting={addingSegmentAfterSegmentId === segment.id}
                  disableToggle={
                    isAddingSegment && addingSegmentAfterSegmentId !== segment.id
                  }
                  onToggleComposer={toggleInsertComposer}
                  onChangeText={(value) => {
                    setNewSegmentDraft(value);
                    if (newSegmentError) {
                      setNewSegmentError(null);
                    }
                  }}
                  onSubmit={() => void handleSubmitInsertComposer(segment.id)}
                  onCancel={handleCancelInsertComposer}
                />
              ) : null}
            </div>
          );
        })}
        <InsertSegmentControls
          targetKey={INSERT_END_TARGET}
          label="Add segment to end"
          isComposerOpen={activeInsertTargetKey === INSERT_END_TARGET}
          draftText={newSegmentDraft}
          errorMessage={newSegmentError}
          isSubmitting={addingSegmentAfterSegmentId === INSERT_END_TARGET}
          disableToggle={
            isAddingSegment && addingSegmentAfterSegmentId !== INSERT_END_TARGET
          }
          onToggleComposer={toggleInsertComposer}
          onChangeText={(value) => {
            setNewSegmentDraft(value);
            if (newSegmentError) {
              setNewSegmentError(null);
            }
          }}
          onSubmit={() => void handleSubmitInsertComposer(INSERT_END_TARGET)}
          onCancel={handleCancelInsertComposer}
        />
      </div>
    </>
  );
}
