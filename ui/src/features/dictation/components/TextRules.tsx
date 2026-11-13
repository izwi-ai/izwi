import { useState, type KeyboardEvent } from "react";
import { Plus, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { normalizeVocabulary } from "@/features/dictation/settings";
import type { DictationReplacement } from "@/features/dictation/types";

const FIELD_CLASS =
  "h-9 rounded-[10px] border-(--border-default) bg-(--surface-panel) text-[13.5px] shadow-none";

const ICON_BUTTON_CLASS =
  "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-(--text-subtle) transition-colors hover:bg-(--bg-surface-2) hover:text-(--text-primary) focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-(--border-strong)";

/** Names and terms as removable chips, with a field to add more. */
export function VocabularyInput({
  terms,
  onChange,
}: {
  terms: string[];
  onChange: (terms: string[]) => void;
}) {
  const [draft, setDraft] = useState("");

  const addDraft = () => {
    const next = normalizeVocabulary([...terms, ...draft.split(",")]);
    setDraft("");
    if (next.length !== terms.length) {
      onChange(next);
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter" || event.key === ",") {
      event.preventDefault();
      addDraft();
    } else if (event.key === "Backspace" && !draft && terms.length > 0) {
      onChange(terms.slice(0, -1));
    }
  };

  return (
    <div className="space-y-2">
      {terms.length > 0 ? (
        <ul aria-label="Vocabulary" className="flex flex-wrap gap-1.5">
          {terms.map((term) => (
            <li
              key={term}
              className="inline-flex items-center gap-1 rounded-full border border-(--border-default) py-0.5 pl-2.5 pr-1 text-[12.5px] text-(--text-secondary)"
            >
              {term}
              <button
                type="button"
                aria-label={`Remove ${term}`}
                onClick={() => onChange(terms.filter((entry) => entry !== term))}
                className="inline-flex h-5 w-5 items-center justify-center rounded-full text-(--text-subtle) transition-colors hover:bg-(--bg-surface-2) hover:text-(--text-primary) focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-(--border-strong)"
              >
                <X className="h-3 w-3" />
              </button>
            </li>
          ))}
        </ul>
      ) : null}
      <Input
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={handleKeyDown}
        onBlur={addDraft}
        aria-label="Add a word or name"
        placeholder="Add a word or name, then press Enter"
        className={`${FIELD_CLASS} max-w-sm`}
      />
    </div>
  );
}

function completeRows(rows: DictationReplacement[]): DictationReplacement[] {
  return rows
    .map((row) => ({ from: row.from.trim(), to: row.to }))
    .filter((row) => row.from);
}

function sameRows(a: DictationReplacement[], b: DictationReplacement[]) {
  return (
    a.length === b.length &&
    a.every((row, index) => row.from === b[index].from && row.to === b[index].to)
  );
}

/**
 * "When I say X, write Y" rows. Rows are saved when a field loses focus;
 * a row with nothing to replace is not saved.
 */
export function ReplacementsEditor({
  replacements,
  onChange,
}: {
  replacements: DictationReplacement[];
  onChange: (replacements: DictationReplacement[]) => void;
}) {
  const [rows, setRows] = useState(replacements);
  const [saved, setSaved] = useState(replacements);

  // Saved rows changed underneath (a refused save, another window): show them.
  if (!sameRows(saved, replacements)) {
    setSaved(replacements);
    setRows(replacements);
  }

  const save = (next: DictationReplacement[]) => {
    const complete = completeRows(next);
    if (!sameRows(complete, replacements)) {
      setSaved(complete);
      onChange(complete);
    }
  };

  const edit = (index: number, change: Partial<DictationReplacement>) =>
    setRows((current) =>
      current.map((row, position) =>
        position === index ? { ...row, ...change } : row,
      ),
    );

  return (
    <div className="space-y-2">
      {rows.length > 0 ? (
        <ul className="space-y-2">
          {rows.map((row, index) => (
            <li key={index} className="flex items-center gap-2">
              <Input
                value={row.from}
                onChange={(event) => edit(index, { from: event.target.value })}
                onBlur={() => save(rows)}
                aria-label={`Replacement ${index + 1}: when I say`}
                placeholder="When I say"
                className={FIELD_CLASS}
              />
              <span aria-hidden="true" className="text-(--text-subtle)">
                →
              </span>
              <Input
                value={row.to}
                onChange={(event) => edit(index, { to: event.target.value })}
                onBlur={() => save(rows)}
                aria-label={`Replacement ${index + 1}: write`}
                placeholder="Write"
                className={FIELD_CLASS}
              />
              <button
                type="button"
                aria-label={`Remove replacement ${index + 1}`}
                onClick={() => {
                  const next = rows.filter((_, position) => position !== index);
                  setRows(next);
                  save(next);
                }}
                className={ICON_BUTTON_CLASS}
              >
                <X className="h-4 w-4" />
              </button>
            </li>
          ))}
        </ul>
      ) : null}
      <Button
        type="button"
        size="sm"
        variant="outline"
        onClick={() => setRows((current) => [...current, { from: "", to: "" }])}
      >
        <Plus />
        Add replacement
      </Button>
    </div>
  );
}
