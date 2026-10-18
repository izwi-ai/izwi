import { useEffect, useRef, useState, type ReactNode } from "react";
import { Search, X } from "lucide-react";

import { cn } from "@/lib/utils";

const SEARCH_DEBOUNCE_MS = 250;

interface SearchFieldProps {
  /** The applied search; typing reports changes after a short pause. */
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  className?: string;
}

export function SearchField({
  value,
  onChange,
  placeholder,
  className,
}: SearchFieldProps) {
  const [draft, setDraft] = useState(value);
  const onChangeRef = useRef(onChange);
  useEffect(() => {
    onChangeRef.current = onChange;
  });

  // Follow outside changes (a cleared filter, a URL change).
  useEffect(() => {
    setDraft(value);
  }, [value]);

  useEffect(() => {
    if (draft === value) {
      return;
    }
    const timer = window.setTimeout(
      () => onChangeRef.current(draft),
      SEARCH_DEBOUNCE_MS,
    );
    return () => window.clearTimeout(timer);
  }, [draft, value]);

  return (
    <label
      className={cn(
        "flex h-[34px] w-full items-center gap-2 rounded-[10px] border border-[var(--border-default)] bg-[var(--surface-panel)] px-2.5 text-[13px] text-[var(--text-subtle)] focus-within:border-[var(--border-strong)] sm:w-[300px]",
        className,
      )}
    >
      <Search aria-hidden="true" className="h-4 w-4 shrink-0" />
      <input
        type="search"
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Escape" && draft) {
            event.preventDefault();
            setDraft("");
            onChange("");
          }
        }}
        placeholder={placeholder}
        aria-label={placeholder.replace(/…$/, "")}
        className="min-w-0 flex-1 bg-transparent text-[var(--text-primary)] outline-none placeholder:text-[var(--text-subtle)] [&::-webkit-search-cancel-button]:hidden"
      />
      {draft ? (
        <button
          type="button"
          onClick={() => {
            setDraft("");
            onChange("");
          }}
          aria-label="Clear search"
          className="rounded p-0.5 hover:text-[var(--text-primary)]"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      ) : null}
    </label>
  );
}

export interface SegmentOption<T extends string> {
  id: T;
  label: string;
  /** Known count for this option; otherwise only the selected one shows a count. */
  count?: number;
}

interface SegmentedFilterProps<T extends string> {
  label: string;
  options: SegmentOption<T>[];
  value: T;
  onChange: (value: T) => void;
  /** Shown beside the selected option (the list total). */
  selectedCount?: number | null;
}

export function SegmentedFilter<T extends string>({
  label,
  options,
  value,
  onChange,
  selectedCount,
}: SegmentedFilterProps<T>) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className="inline-flex gap-0.5 overflow-x-auto rounded-[10px] border border-[var(--border-subtle)] bg-[var(--surface-panel)] p-[3px]"
    >
      {options.map((option) => {
        const selected = option.id === value;
        const count =
          option.count ?? (selected ? (selectedCount ?? undefined) : undefined);
        return (
          <button
            key={option.id}
            type="button"
            role="radio"
            aria-checked={selected}
            onClick={() => onChange(option.id)}
            className={cn(
              "inline-flex h-[26px] items-center gap-1.5 whitespace-nowrap rounded-[7px] px-2.5 text-[12.5px] font-medium transition-colors",
              selected
                ? "bg-[var(--surface-raised)] text-[var(--text-primary)]"
                : "text-[var(--text-muted)] hover:text-[var(--text-primary)]",
            )}
          >
            {option.label}
            {count != null ? " " : null}
            {count != null ? (
              <span className="text-[11px] text-[var(--text-subtle)]">
                {count}
              </span>
            ) : null}
          </button>
        );
      })}
    </div>
  );
}

export function HistoryToolbar({ children }: { children: ReactNode }) {
  return (
    <div className="mb-3.5 flex flex-wrap items-center gap-2">{children}</div>
  );
}
