import type { ButtonHTMLAttributes, ReactNode, Ref } from "react";
import { ChevronLeft, Loader2 } from "lucide-react";

import { pageStateButtonClass } from "@/components/ui/page-state";
import { cn } from "@/lib/utils";
import { isMacPlatform } from "@/lib/platform";
import { APP_ICON_URL } from "@/shared/config/runtime";

/** "Mac" in copy on macOS, else "computer". */
export function deviceNoun(): string {
  return isMacPlatform() ? "Mac" : "computer";
}

export const primaryButtonClass = cn(pageStateButtonClass.primary, "h-11 w-full gap-2");

/** An inline action inside a sentence ("Choose models myself"). */
export const linkButtonClass =
  "rounded-sm text-[var(--text-primary)] underline decoration-[var(--border-strong)] underline-offset-[3px] hover:text-[var(--text-muted)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--border-strong)]";

export function PrimaryButton({
  busy = false,
  className,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { busy?: boolean }) {
  return (
    <button
      type="button"
      {...props}
      disabled={props.disabled || busy}
      aria-busy={busy || undefined}
      className={cn(primaryButtonClass, className)}
    >
      {busy ? <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" /> : null}
      {children}
    </button>
  );
}

export function BackButton({
  onClick,
  children = "Back",
}: {
  onClick: () => void;
  children?: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="-ml-1 flex w-fit items-center gap-1 rounded-md px-1 py-0.5 text-[13px] text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--border-strong)]"
    >
      <ChevronLeft aria-hidden="true" className="h-3.5 w-3.5" />
      {children}
    </button>
  );
}

/**
 * The full window first run fills: the brand, three step bars and Skip
 * setup, over one centred stage. No sidebar behind it.
 */
export function OnboardingFrame({
  step,
  onSkip,
  skipping = false,
  children,
}: {
  /** 1–3; no bars when absent (the engine isn't responding). */
  step?: 1 | 2 | 3;
  onSkip?: () => void;
  skipping?: boolean;
  children: ReactNode;
}) {
  return (
    <div className="flex h-dvh flex-col bg-[var(--surface-canvas)] text-[var(--text-primary)]">
      <header className="grid h-14 shrink-0 grid-cols-3 items-center px-6">
        <div className="flex items-center gap-2.5">
          <img src={APP_ICON_URL} alt="" className="h-6 w-6 rounded-md" />
          <span className="text-[13.5px] font-semibold tracking-tight">Izwi</span>
        </div>
        <div className="flex items-center justify-center">
          {step ? (
            <div
              role="img"
              aria-label={`Step ${step} of 3`}
              className="flex items-center gap-1.5"
            >
              {[1, 2, 3].map((bar) => (
                <span
                  key={bar}
                  className={cn(
                    "h-1 w-6 rounded-full",
                    bar < step && "bg-[var(--text-subtle)]",
                    bar === step && "bg-[var(--accent-solid)]",
                    bar > step && "bg-[var(--surface-raised)]",
                  )}
                />
              ))}
            </div>
          ) : null}
        </div>
        <div className="flex justify-end">
          {onSkip ? (
            <button
              type="button"
              onClick={onSkip}
              disabled={skipping}
              className="rounded-md px-2 py-1 text-[13px] text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--border-strong)] disabled:opacity-50"
            >
              Skip setup
            </button>
          ) : null}
        </div>
      </header>
      <main className="flex min-h-0 flex-1 flex-col overflow-y-auto px-6">
        {children}
      </main>
    </div>
  );
}

/**
 * A centred step. Spacing tightens in short windows so the primary action
 * stays in view at the 680px minimum; past that the stage scrolls.
 */
export function OnboardingStage({
  width = "narrow",
  children,
}: {
  width?: "narrow" | "wide";
  children: ReactNode;
}) {
  return (
    <div
      className={cn(
        "m-auto flex w-full flex-col pb-14 pt-6 [@media(max-height:760px)]:gap-5 [@media(max-height:760px)]:pb-6 [@media(max-height:760px)]:pt-2",
        width === "narrow" ? "max-w-[440px] gap-7" : "max-w-[480px] gap-6",
      )}
    >
      {children}
    </div>
  );
}

export function StageHeading({
  title,
  description,
  headingRef,
  before,
  after,
}: {
  title: string;
  description?: ReactNode;
  headingRef?: Ref<HTMLHeadingElement>;
  before?: ReactNode;
  after?: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      {before}
      <h1
        ref={headingRef}
        tabIndex={-1}
        className="text-2xl font-medium tracking-[-0.02em] text-[var(--text-primary)] focus:outline-none"
      >
        {title}
      </h1>
      {description ? (
        <p className="text-sm leading-relaxed text-[var(--text-muted)]">
          {description}
        </p>
      ) : null}
      {after}
    </div>
  );
}
