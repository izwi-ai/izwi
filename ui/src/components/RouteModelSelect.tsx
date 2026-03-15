import { useEffect, useMemo, useRef, useState, type ComponentProps } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown } from "lucide-react";

import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/ui/status-badge";
import { cn } from "@/lib/utils";

interface RouteModelSelectOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

interface RouteModelSelectProps {
  value: string | null;
  options: RouteModelSelectOption[];
  onSelect?: (value: string) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
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

export function RouteModelSelect({
  value,
  options,
  onSelect,
  placeholder = "Select model",
  className,
  disabled = false,
}: RouteModelSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const selectedOption = useMemo(
    () => options.find((option) => option.value === value) ?? null,
    [options, value],
  );

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

  return (
    <div ref={containerRef} className={cn("relative", className)}>
      <Button
        type="button"
        variant="outline"
        onClick={() => {
          if (!disabled && options.length > 0) {
            setIsOpen((current) => !current);
          }
        }}
        disabled={disabled || options.length === 0}
        className={cn(
          "h-11 w-full justify-between rounded-xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3.5 font-normal shadow-none transition-colors hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-1)]",
          selectedOption?.isReady &&
            "border-primary/20 bg-primary/5 hover:border-primary/30 hover:bg-primary/10",
        )}
      >
        <span className="min-w-0 flex-1 truncate text-left text-sm font-medium text-[var(--text-primary)]">
          {selectedOption?.label || placeholder}
        </span>
        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 shrink-0 text-[var(--text-muted)] transition-transform",
            isOpen && "rotate-180",
          )}
        />
      </Button>

      <AnimatePresence>
        {isOpen ? (
          <motion.div
            initial={{ opacity: 0, y: 6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.98 }}
            transition={{ duration: 0.16 }}
            className="absolute inset-x-0 top-full z-[90] mt-2 rounded-xl border border-[var(--border-strong)] bg-[var(--bg-surface-0)] p-1.5 shadow-xl"
          >
            <div className="max-h-72 overflow-y-auto">
              {options.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => {
                    onSelect?.(option.value);
                    setIsOpen(false);
                  }}
                  className={cn(
                    "relative flex w-full items-center rounded-lg px-2.5 py-2 text-left transition-colors hover:bg-[var(--bg-surface-1)]",
                    selectedOption?.value === option.value &&
                      "bg-[var(--bg-surface-1)]",
                  )}
                >
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-medium text-[var(--text-primary)]">
                      {option.label}
                    </div>
                    <StatusBadge
                      tone={getStatusTone(option)}
                      className="mt-1"
                    >
                      {option.statusLabel}
                    </StatusBadge>
                  </div>
                </button>
              ))}
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
