import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Check,
  ChevronDown,
  Music4,
  Play,
  Sparkles,
  Square,
  Waves,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import type { VoicePickerItem } from "@/components/VoicePicker";

type VoiceMode = "saved" | "built_in";

interface VoiceSelectProps {
  voiceMode: VoiceMode;
  onVoiceModeChange: (mode: VoiceMode) => void;
  savedVoiceItems: VoicePickerItem[];
  builtInVoiceItems: VoicePickerItem[];
  selectedItem: VoicePickerItem | null;
  savedVoicesLoading?: boolean;
  savedVoicesError?: string | null;
  savedEnabled?: boolean;
  builtInEnabled?: boolean;
  disabled?: boolean;
  modelLabel?: string | null;
}

const MODE_COPY: Record<
  VoiceMode,
  {
    label: string;
    triggerLabel: string;
    emptyTitle: string;
    emptyDescription: string;
  }
> = {
  built_in: {
    label: "Built-in Voices",
    triggerLabel: "Built-in",
    emptyTitle: "No built-in voices",
    emptyDescription:
      "This model does not expose selectable built-in voices on this route.",
  },
  saved: {
    label: "My Voices",
    triggerLabel: "My voices",
    emptyTitle: "No saved voices",
    emptyDescription: "Save a designed or cloned voice to reuse it here.",
  },
};

function itemInitial(item: VoicePickerItem | null, fallback: string): string {
  const firstCharacter = item?.name?.trim()?.charAt(0);
  return (firstCharacter || fallback).toUpperCase();
}

function itemMatchesSearch(item: VoicePickerItem, query: string): boolean {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) {
    return true;
  }

  return [item.name, item.categoryLabel, item.description, ...(item.meta ?? [])]
    .filter((value): value is string => Boolean(value))
    .some((value) => value.toLowerCase().includes(normalizedQuery));
}

export function VoiceSelect({
  voiceMode,
  onVoiceModeChange,
  savedVoiceItems,
  builtInVoiceItems,
  selectedItem,
  savedVoicesLoading = false,
  savedVoicesError = null,
  savedEnabled = true,
  builtInEnabled = true,
  disabled = false,
  modelLabel = null,
}: VoiceSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [playingId, setPlayingId] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const availableModes = useMemo(() => {
    const modes: VoiceMode[] = [];
    if (builtInEnabled) {
      modes.push("built_in");
    }
    if (savedEnabled) {
      modes.push("saved");
    }
    return modes;
  }, [builtInEnabled, savedEnabled]);

  const activeMode =
    availableModes.find((mode) => mode === voiceMode) ??
    availableModes[0] ??
    "built_in";

  const activeItems =
    activeMode === "saved" ? savedVoiceItems : builtInVoiceItems;
  const filteredItems = useMemo(
    () => activeItems.filter((item) => itemMatchesSearch(item, search)),
    [activeItems, search],
  );
  const canOpen = !disabled && availableModes.length > 0;
  const showSearch = activeItems.length > 6 || search.trim().length > 0;

  useEffect(() => {
    const handlePointerDown = (event: MouseEvent) => {
      if (
        containerRef.current &&
        event.target instanceof Node &&
        !containerRef.current.contains(event.target)
      ) {
        setIsOpen(false);
      }
    };

    window.addEventListener("mousedown", handlePointerDown);
    return () => window.removeEventListener("mousedown", handlePointerDown);
  }, []);

  useEffect(() => {
    const element = audioRef.current;
    if (!element) {
      return;
    }

    const handleEnded = () => setPlayingId(null);
    element.addEventListener("ended", handleEnded);
    return () => element.removeEventListener("ended", handleEnded);
  }, []);

  useEffect(() => {
    if (!isOpen && playingId) {
      audioRef.current?.pause();
      setPlayingId(null);
    }
  }, [isOpen, playingId]);

  useEffect(() => {
    setSearch("");
  }, [activeMode, isOpen]);

  const togglePreview = (
    event: ReactMouseEvent<HTMLButtonElement>,
    url: string | null | undefined,
    id: string,
  ) => {
    event.stopPropagation();
    if (!url || !audioRef.current) {
      return;
    }

    if (playingId === id) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setPlayingId(null);
      return;
    }

    audioRef.current.src = url;
    audioRef.current.currentTime = 0;
    audioRef.current.play().catch(() => {});
    setPlayingId(id);
  };

  const triggerSubtitle = selectedItem
    ? [selectedItem.categoryLabel, selectedItem.meta?.[0]]
        .filter((value): value is string => Boolean(value))
        .join(" · ")
    : canOpen
      ? modelLabel
        ? `Available for ${modelLabel}`
        : "Choose a model-compatible voice"
      : modelLabel
        ? `${modelLabel} does not expose voices here`
        : "Select a compatible model first";

  const renderEmptyState = () => {
    const isSearchEmpty = search.trim().length === 0;
    const emptyTitle =
      !isSearchEmpty && filteredItems.length === 0
        ? "No matching voices"
        : activeMode === "saved" && savedVoicesLoading
          ? "Loading saved voices"
          : MODE_COPY[activeMode].emptyTitle;
    const emptyDescription =
      !isSearchEmpty && filteredItems.length === 0
        ? "Try a different name, transcript fragment, or language."
        : activeMode === "saved" && savedVoicesLoading
          ? "Fetching your reusable voice library."
          : MODE_COPY[activeMode].emptyDescription;

    return (
      <div className="flex flex-col items-center justify-center px-5 py-8 text-center">
        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3">
          <Music4 className="h-4 w-4 text-[var(--text-muted)]" />
        </div>
        <div className="mt-3 space-y-1">
          <div className="text-sm font-semibold text-[var(--text-primary)]">
            {emptyTitle}
          </div>
          <div className="max-w-[240px] text-xs leading-relaxed text-[var(--text-muted)]">
            {emptyDescription}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div ref={containerRef} className="relative w-full">
      <audio ref={audioRef} className="hidden" />

      <button
        type="button"
        onClick={() => {
          if (!canOpen) {
            return;
          }
          setIsOpen((current) => !current);
        }}
        disabled={!canOpen}
        className={cn(
          "flex w-full items-center justify-between gap-3 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3.5 py-2.5 text-left shadow-none transition-[border-color,box-shadow,background-color] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/35 focus-visible:border-ring/50",
          canOpen
            ? "hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-1)]"
            : "cursor-not-allowed bg-[var(--bg-surface-1)]/60 text-[var(--text-muted)]",
          isOpen && "border-ring/50 ring-2 ring-ring/35",
        )}
      >
        <div className="speaker-avatar flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-medium">
          {itemInitial(selectedItem, activeMode === "saved" ? "S" : "B")}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-medium text-[var(--text-primary)]">
              {selectedItem?.name ?? (canOpen ? "Select a voice" : "No voices available")}
            </span>
            <span className="rounded-md bg-[var(--bg-surface-2)] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
              {MODE_COPY[activeMode].triggerLabel}
            </span>
          </div>
          <div className="mt-0.5 truncate text-[11px] text-[var(--text-muted)]">
            {triggerSubtitle}
          </div>
        </div>
        <ChevronDown
          className={cn(
            "h-4 w-4 shrink-0 text-[var(--text-muted)] transition-transform duration-200",
            isOpen && "rotate-180",
          )}
        />
      </button>

      <AnimatePresence>
        {isOpen ? (
          <motion.div
            initial={{ opacity: 0, y: 8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.98 }}
            transition={{ duration: 0.15, ease: "easeOut" }}
            className="absolute inset-x-0 top-[calc(100%+8px)] z-50 overflow-hidden rounded-2xl border border-[var(--border-strong)] bg-[var(--bg-surface-0)] shadow-[0_16px_48px_-18px_rgba(0,0,0,0.35)]"
          >
            <div className="space-y-3 p-2.5">
              {availableModes.length > 1 ? (
                <div className="grid grid-cols-2 gap-1 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-1">
                  {availableModes.map((mode) => {
                    const Icon = mode === "saved" ? Waves : Sparkles;
                    return (
                      <button
                        key={mode}
                        type="button"
                        onClick={() => onVoiceModeChange(mode)}
                        className={cn(
                          "inline-flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-semibold transition-colors",
                          activeMode === mode
                            ? "bg-[var(--bg-surface-0)] text-[var(--text-primary)] shadow-sm"
                            : "text-[var(--text-muted)] hover:bg-[var(--bg-surface-0)] hover:text-[var(--text-primary)]",
                        )}
                      >
                        <Icon className="h-4 w-4" />
                        {MODE_COPY[mode].label}
                      </button>
                    );
                  })}
                </div>
              ) : null}

              <div className="flex items-center justify-between gap-3 px-1">
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--text-muted)]">
                    Available for
                  </div>
                  <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
                    {modelLabel ?? "No model selected"}
                  </div>
                </div>
                {canOpen ? (
                  <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-1 text-[10px] font-medium text-[var(--text-muted)]">
                    {activeItems.length} voices
                  </span>
                ) : null}
              </div>

              {showSearch ? (
                <Input
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Search by name, transcript, or language"
                  className="bg-[var(--bg-surface-0)]"
                />
              ) : null}

              {activeMode === "saved" && savedVoicesError ? (
                <div className="rounded-xl border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs text-destructive">
                  {savedVoicesError}
                </div>
              ) : null}
            </div>

            <div className="max-h-[340px] overflow-y-auto px-2.5 pb-2.5">
              <div className="space-y-1.5">
                {filteredItems.length === 0
                  ? renderEmptyState()
                  : filteredItems.map((item) => (
                      <div
                        key={item.id}
                        className={cn(
                          "flex items-start gap-3 rounded-xl border px-3 py-2.5 transition-colors",
                          item.selected
                            ? "border-primary/30 bg-primary/5"
                            : "border-transparent bg-[var(--bg-surface-0)] hover:border-[var(--border-muted)] hover:bg-[var(--bg-surface-1)]",
                        )}
                      >
                        <button
                          type="button"
                          onClick={() => {
                            item.onSelect?.();
                            setIsOpen(false);
                          }}
                          className="flex min-w-0 flex-1 items-start gap-3 rounded-lg text-left"
                        >
                          <div className="speaker-avatar mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-medium">
                            {itemInitial(item, activeMode === "saved" ? "S" : "B")}
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-2">
                              <span className="truncate text-sm font-medium text-[var(--text-primary)]">
                                {item.name}
                              </span>
                              {item.selected ? (
                                <Check className="h-3.5 w-3.5 shrink-0 text-primary" />
                              ) : null}
                            </div>
                            <div className="mt-0.5 truncate text-[11px] text-[var(--text-muted)]">
                              {item.categoryLabel}
                            </div>

                            {item.description ? (
                              <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-[var(--text-secondary)]">
                                {item.description}
                              </p>
                            ) : item.previewMessage ? (
                              <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-[var(--text-muted)]">
                                {item.previewMessage}
                              </p>
                            ) : null}

                            {item.meta && item.meta.length > 0 ? (
                              <div className="mt-2 flex flex-wrap gap-1.5">
                                {item.meta.map((meta) => (
                                  <span
                                    key={`${item.id}-${meta}`}
                                    className="rounded-md bg-[var(--bg-surface-2)] px-2 py-0.5 text-[10px] font-medium text-[var(--text-muted)]"
                                  >
                                    {meta}
                                  </span>
                                ))}
                              </div>
                            ) : null}
                          </div>
                        </button>

                        {item.previewUrl ? (
                          <button
                            type="button"
                            onClick={(event) =>
                              togglePreview(event, item.previewUrl, item.id)
                            }
                            className={cn(
                              "mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full border bg-[var(--bg-surface-0)] transition-colors",
                              playingId === item.id
                                ? "border-primary text-primary shadow-sm"
                                : "border-[var(--border-muted)] text-[var(--text-muted)] hover:bg-[var(--bg-surface-1)]",
                            )}
                            aria-label={
                              playingId === item.id
                                ? `Stop preview for ${item.name}`
                                : `Play preview for ${item.name}`
                            }
                          >
                            {playingId === item.id ? (
                              <Square className="h-3.5 w-3.5 fill-current" />
                            ) : (
                              <Play className="ml-0.5 h-3.5 w-3.5 fill-current" />
                            )}
                          </button>
                        ) : null}
                      </div>
                    ))}
              </div>
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
