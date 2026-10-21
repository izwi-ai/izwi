import { useEffect, useMemo, useState, type KeyboardEvent } from "react";
import { useNavigate } from "react-router";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import {
  AudioWaveform,
  FilePlus2,
  Monitor,
  Moon,
  Search,
  Sun,
  UserRoundPlus,
  Users,
  Wand2,
  type LucideIcon,
} from "lucide-react";

import { trackThemePreferenceChanged } from "@/app/analytics/events";
import { NAV_GROUPS, SETTINGS_NAV_ITEM } from "@/app/layouts/navigation";
import { useTheme } from "@/app/providers/ThemeProvider";
import { cn } from "@/lib/utils";

interface Command {
  id: string;
  group: string;
  label: string;
  icon: LucideIcon;
  keywords?: string;
  run: () => void;
}

function matches(command: Command, query: string): boolean {
  const haystack =
    `${command.label} ${command.group} ${command.keywords ?? ""}`.toLowerCase();
  return query
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((term) => haystack.includes(term));
}

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/** ⌘K: jump to any page, start new work or switch theme. */
export function CommandPalette({ open, onOpenChange }: CommandPaletteProps) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-80 bg-background/60 backdrop-blur-[2px]" />
        <DialogPrimitive.Content
          aria-describedby={undefined}
          className="fixed left-1/2 top-[14vh] z-80 w-[calc(100vw-2rem)] max-w-xl -translate-x-1/2 overflow-hidden rounded-2xl border border-(--border-default) bg-(--surface-panel) shadow-[0_24px_70px_-24px_rgba(0,0,0,0.6)]"
        >
          <DialogPrimitive.Title className="sr-only">
            Command palette
          </DialogPrimitive.Title>
          {/* Remounted per open, so the query and selection start fresh. */}
          {open ? (
            <CommandPaletteBody onDone={() => onOpenChange(false)} />
          ) : null}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

function CommandPaletteBody({ onDone }: { onDone: () => void }) {
  const navigate = useNavigate();
  const { setThemePreference } = useTheme();
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);

  const commands = useMemo<Command[]>(() => {
    const go = (path: string) => () => navigate(path);
    const theme = (preference: "light" | "dark" | "system") => () => {
      setThemePreference(preference);
      void trackThemePreferenceChanged(preference);
    };
    return [
      ...[...NAV_GROUPS.flatMap((group) => group.items), SETTINGS_NAV_ITEM].map(
        (item) => ({
          id: `go-${item.id}`,
          group: "Go to",
          label: item.label,
          icon: item.icon,
          run: go(item.path),
        }),
      ),
      {
        id: "new-transcript",
        group: "Create",
        label: "New transcript",
        icon: FilePlus2,
        keywords: "transcribe audio upload",
        run: go("/transcription?create=transcription"),
      },
      {
        id: "new-diarization",
        group: "Create",
        label: "New diarization",
        icon: Users,
        keywords: "speakers transcribe",
        run: go("/transcription?create=diarization"),
      },
      {
        id: "new-speech",
        group: "Create",
        label: "New speech",
        icon: AudioWaveform,
        keywords: "text to speech tts generate",
        run: go("/text-to-speech?create=speech"),
      },
      {
        id: "clone-voice",
        group: "Create",
        label: "Clone a voice",
        icon: UserRoundPlus,
        keywords: "voice cloning reference audio",
        run: go("/voice-cloning"),
      },
      {
        id: "design-voice",
        group: "Create",
        label: "Design a voice",
        icon: Wand2,
        keywords: "voice design description",
        run: go("/voice-design"),
      },
      {
        id: "theme-light",
        group: "Theme",
        label: "Light theme",
        icon: Sun,
        keywords: "appearance",
        run: theme("light"),
      },
      {
        id: "theme-dark",
        group: "Theme",
        label: "Dark theme",
        icon: Moon,
        keywords: "appearance",
        run: theme("dark"),
      },
      {
        id: "theme-system",
        group: "Theme",
        label: "Match system theme",
        icon: Monitor,
        keywords: "appearance auto",
        run: theme("system"),
      },
    ];
  }, [navigate, setThemePreference]);

  const visible = useMemo(
    () => commands.filter((command) => matches(command, query)),
    [commands, query],
  );
  const active = Math.min(activeIndex, Math.max(visible.length - 1, 0));
  const activeId = visible[active]?.id;

  useEffect(() => {
    if (activeId) {
      document
        .getElementById(`command-option-${activeId}`)
        ?.scrollIntoView?.({ block: "nearest" });
    }
  }, [activeId]);

  const runCommand = (command: Command | undefined) => {
    if (!command) {
      return;
    }
    onDone();
    command.run();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (visible.length === 0) {
        return;
      }
      const step = event.key === "ArrowDown" ? 1 : -1;
      setActiveIndex((active + step + visible.length) % visible.length);
    } else if (event.key === "Enter") {
      event.preventDefault();
      runCommand(visible[active]);
    }
  };

  const optionId = (command: Command) => `command-option-${command.id}`;

  return (
    <>
      <div className="flex items-center gap-2.5 border-b border-(--border-subtle) px-4">
        <Search className="h-4 w-4 shrink-0 text-(--text-subtle)" />
        <input
          autoFocus
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setActiveIndex(0);
          }}
          onKeyDown={handleKeyDown}
          placeholder="Search pages and actions…"
          role="combobox"
          aria-expanded="true"
          aria-controls="command-palette-list"
          aria-activedescendant={
            visible[active] ? optionId(visible[active]) : undefined
          }
          aria-label="Search pages and actions"
          className="h-12 flex-1 bg-transparent text-sm text-(--text-primary) outline-hidden placeholder:text-(--text-subtle)"
        />
      </div>
      <div
        id="command-palette-list"
        role="listbox"
        aria-label="Commands"
        className="max-h-[min(24rem,60vh)] overflow-y-auto p-1.5"
      >
        {visible.length === 0 ? (
          <p className="px-3 py-6 text-center text-sm text-(--text-muted)">
            No matching pages or actions.
          </p>
        ) : (
          visible.map((command, index) => {
            const startsGroup =
              index === 0 || visible[index - 1].group !== command.group;
            return (
              <div key={command.id}>
                {startsGroup ? (
                  <div
                    aria-hidden="true"
                    className="px-2.5 pb-1 pt-2 text-[11px] font-medium text-(--text-subtle)"
                  >
                    {command.group}
                  </div>
                ) : null}
                <div
                  id={optionId(command)}
                  role="option"
                  aria-selected={index === active}
                  onMouseMove={() => setActiveIndex(index)}
                  onClick={() => runCommand(command)}
                  className={cn(
                    "flex h-9 cursor-pointer items-center gap-2.5 rounded-lg px-2.5 text-[13.5px] text-(--text-secondary)",
                    index === active &&
                      "bg-(--sidebar-link-active-bg) text-(--text-primary)",
                  )}
                >
                  <command.icon className="h-4 w-4 shrink-0 text-(--text-muted)" />
                  {command.label}
                </div>
              </div>
            );
          })
        )}
      </div>
    </>
  );
}
