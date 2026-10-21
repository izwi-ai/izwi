import * as SelectPrimitive from "@radix-ui/react-select";
import { Check, ChevronDown } from "lucide-react";

import { Select, SelectContent } from "@/components/ui/select";
import type { ModelOption } from "@/features/chat/playground/support";
import { cn } from "@/lib/utils";

/** Not a model: picking it opens the model manager. */
const MANAGE_VALUE = "__manage_chat_models__";

const ON_DEVICE = new Set<ModelOption["status"]>([
  "downloaded",
  "loading",
  "ready",
]);

const itemClass =
  "relative flex h-8 cursor-default select-none items-center rounded-md pl-2.5 pr-8 text-[13.5px] outline-hidden data-disabled:pointer-events-none data-disabled:opacity-50 data-highlighted:bg-(--accent-soft) data-[state=checked]:bg-(--accent-soft)";

interface ChatModelPickerProps {
  value: string | null;
  options: ModelOption[];
  onSelect: (variant: string) => void;
  /** Opens the chat model manager. */
  onManage: () => void;
  disabled?: boolean;
  menuPlacement: "top" | "bottom";
}

/**
 * The composer's model pill: the chat models on this device, by name, with
 * a check on the one in use. The rest are in the model manager.
 */
export function ChatModelPicker({
  value,
  options,
  onSelect,
  onManage,
  disabled = false,
  menuPlacement,
}: ChatModelPickerProps) {
  const selected = options.find((option) => option.value === value) ?? null;
  const shown = options.filter(
    (option) =>
      option.value === value ||
      option.isReady ||
      (option.status !== undefined && ON_DEVICE.has(option.status)),
  );

  return (
    <Select
      value={value ?? ""}
      onValueChange={(next) =>
        next === MANAGE_VALUE ? onManage() : onSelect(next)
      }
      disabled={disabled}
    >
      <SelectPrimitive.Trigger
        aria-label="Chat model"
        className="inline-flex h-9 min-w-0 max-w-[16rem] items-center gap-1.5 rounded-full border border-(--border-default) px-3.5 text-[13.5px] font-medium text-(--text-primary) outline-hidden transition-colors hover:border-(--border-strong) focus-visible:ring-2 focus-visible:ring-(--border-strong) disabled:cursor-not-allowed disabled:opacity-50 data-[state=open]:border-(--border-strong) data-[state=open]:bg-(--accent-soft)"
      >
        <span className="truncate">{selected?.label ?? "Choose a model"}</span>
        <ChevronDown
          aria-hidden="true"
          className="h-4 w-4 shrink-0 text-(--text-muted)"
        />
      </SelectPrimitive.Trigger>
      <SelectContent
        position="popper"
        side={menuPlacement}
        align="start"
        sideOffset={8}
        className="z-90 w-66 rounded-[12px] border-(--border-default) bg-(--surface-panel) p-1"
      >
        {shown.map((option) => (
          <SelectPrimitive.Item
            key={option.value}
            value={option.value}
            textValue={option.label}
            className={itemClass}
          >
            <SelectPrimitive.ItemText>
              <span className="block truncate">{option.label}</span>
            </SelectPrimitive.ItemText>
            <SelectPrimitive.ItemIndicator className="absolute right-2.5">
              <Check
                aria-hidden="true"
                strokeWidth={2.5}
                className="h-[15px] w-[15px]"
              />
            </SelectPrimitive.ItemIndicator>
          </SelectPrimitive.Item>
        ))}
        {shown.length > 0 && (
          <SelectPrimitive.Separator className="my-1 h-px bg-(--border-subtle)" />
        )}
        <SelectPrimitive.Item
          value={MANAGE_VALUE}
          textValue="Manage chat models…"
          className={cn(itemClass, "text-[13px] text-(--text-muted)")}
        >
          <SelectPrimitive.ItemText>Manage chat models…</SelectPrimitive.ItemText>
        </SelectPrimitive.Item>
      </SelectContent>
    </Select>
  );
}
