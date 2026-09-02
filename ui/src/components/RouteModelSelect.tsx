import { useId, useMemo, type ComponentProps } from "react";
import * as SelectPrimitive from "@radix-ui/react-select";
import {
  AlertTriangle,
  ArrowDownToLine,
  Check,
  CheckCircle2,
  CircleDashed,
  Loader2,
} from "lucide-react";

import {
  Select,
  SelectContent,
  SelectTrigger,
} from "@/components/ui/select";
import { StatusBadge } from "@/components/ui/status-badge";
import { cn } from "@/lib/utils";

interface RouteModelSelectOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
  disabled?: boolean;
}

interface RouteModelSelectProps {
  value: string | null;
  options: RouteModelSelectOption[];
  onSelect?: (value: string) => void;
  placeholder?: string;
  className?: string;
  triggerClassName?: string;
  disabled?: boolean;
  menuPlacement?: "top" | "bottom";
  "aria-label"?: string;
  description?: string;
}

function getStatusTone(
  option: RouteModelSelectOption,
): ComponentProps<typeof StatusBadge>["tone"] {
  const normalizedStatus = option.statusLabel.toLowerCase();

  if (option.isReady) {
    return "success";
  }
  if (
    normalizedStatus.includes("downloading") ||
    normalizedStatus.includes("loading")
  ) {
    return "warning";
  }
  if (normalizedStatus.includes("error")) {
    return "danger";
  }
  return "neutral";
}

function getStatusPresentation(option: RouteModelSelectOption): {
  icon: typeof CheckCircle2;
  className: string;
} {
  const normalizedStatus = option.statusLabel.toLowerCase();

  if (option.isReady) {
    return {
      icon: CheckCircle2,
      className: "text-[var(--status-positive-text)]",
    };
  }
  if (normalizedStatus.includes("downloaded")) {
    return {
      icon: ArrowDownToLine,
      className: "text-[var(--status-info-text)]",
    };
  }
  if (
    normalizedStatus.includes("downloading") ||
    normalizedStatus.includes("loading")
  ) {
    return {
      icon: Loader2,
      className: "text-[var(--status-warning-text)]",
    };
  }
  if (normalizedStatus.includes("error")) {
    return {
      icon: AlertTriangle,
      className: "text-[var(--danger-text)]",
    };
  }

  return {
    icon: CircleDashed,
    className: "text-[var(--text-muted)]",
  };
}

export function RouteModelSelect({
  value,
  options,
  onSelect,
  placeholder = "Select model",
  className,
  triggerClassName,
  disabled = false,
  menuPlacement = "bottom",
  "aria-label": ariaLabel,
  description,
}: RouteModelSelectProps) {
  const descriptionId = useId();

  const selectedOption = useMemo(
    () => options.find((option) => option.value === value) ?? null,
    [options, value],
  );
  const selectedStatus = selectedOption
    ? getStatusPresentation(selectedOption)
    : null;

  const accessibleDescription =
    description ??
    (selectedOption
      ? `${selectedOption.label}. Status: ${selectedOption.statusLabel}.`
      : `${placeholder}.`);

  return (
    <div className={cn("relative", className)}>
      <Select
        value={value ?? ""}
        onValueChange={(nextValue) => onSelect?.(nextValue)}
        disabled={disabled || options.length === 0}
      >
        <SelectTrigger
          aria-label={ariaLabel ?? selectedOption?.label ?? placeholder}
          aria-describedby={descriptionId}
          className={cn(
            "w-full rounded-[var(--radius-md)] border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3.5 font-normal text-[var(--text-primary)] shadow-[var(--shadow-soft)] transition-[border-color,background-color,box-shadow] hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-1)] data-[state=open]:border-ring/50 data-[state=open]:ring-2 data-[state=open]:ring-ring/35",
            triggerClassName ?? "h-10",
          )}
        >
          <div className="min-w-0 flex flex-1 items-center gap-2">
            {selectedStatus ? (
              <selectedStatus.icon
                aria-hidden="true"
                className={cn(
                  "h-3.5 w-3.5 shrink-0",
                  selectedStatus.className,
                  selectedOption?.statusLabel.toLowerCase().includes("loading") &&
                    "animate-spin",
                )}
              />
            ) : null}
            <span
              className="min-w-0 flex-1 truncate text-left text-sm font-medium text-[var(--text-primary)]"
              title={selectedOption?.label || placeholder}
            >
              {selectedOption?.label || placeholder}
            </span>
          </div>
        </SelectTrigger>

        <SelectContent
          position="popper"
          side={menuPlacement}
          sideOffset={8}
          className="z-[90] max-h-72 min-w-[18rem] max-w-[min(36rem,calc(100vw-2rem))] p-1.5"
        >
          {options.map((option) => (
            <SelectPrimitive.Item
              key={option.value}
              value={option.value}
              disabled={option.disabled}
              textValue={option.label}
              className={cn(
                "relative flex min-w-[18rem] cursor-default select-none items-start rounded-[var(--radius-sm)] py-2.5 pl-3 pr-9 text-left outline-none transition-colors focus:bg-[var(--bg-surface-1)] focus:text-[var(--text-primary)] data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
                selectedOption?.value === option.value &&
                  "bg-[var(--bg-surface-1)]",
              )}
              title={option.label}
            >
              <SelectPrimitive.ItemText>
                <span className="sr-only">
                  {option.label}. Status: {option.statusLabel}.
                </span>
              </SelectPrimitive.ItemText>
              <div aria-hidden="true" className="flex w-full min-w-0 items-start gap-3">
                <div className="flex min-w-0 flex-1 items-start gap-2.5">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
                    {(() => {
                      const status = getStatusPresentation(option);
                      const StatusIcon = status.icon;
                      return (
                        <StatusIcon
                          aria-hidden="true"
                          className={cn(
                            "h-4 w-4",
                            status.className,
                            option.statusLabel.toLowerCase().includes("loading") &&
                              "animate-spin",
                          )}
                        />
                      );
                    })()}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-medium leading-5 text-[var(--text-primary)] break-words">
                      {option.label}
                    </div>
                    <div className="mt-1 text-xs text-[var(--text-muted)]">
                      {option.statusLabel}
                    </div>
                  </div>
                </div>
                {selectedOption?.value === option.value ? (
                  <div aria-hidden="true" className="flex shrink-0 items-center gap-2 pl-2">
                    <StatusBadge
                      tone={getStatusTone(option)}
                      className="px-2 py-0.5 text-[9px] tracking-[0.14em]"
                    >
                      Current
                    </StatusBadge>
                  </div>
                ) : null}
              </div>
              <SelectPrimitive.ItemIndicator className="absolute right-3 top-3.5">
                <Check aria-hidden="true" className="h-4 w-4" />
              </SelectPrimitive.ItemIndicator>
            </SelectPrimitive.Item>
          ))}
        </SelectContent>
      </Select>
      <span id={descriptionId} className="sr-only">
        {accessibleDescription}
      </span>
    </div>
  );
}
