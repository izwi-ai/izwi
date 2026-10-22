import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, TriangleAlert } from "lucide-react";

import { api } from "@/api";
import {
  OnboardingStage,
  PrimaryButton,
  StageHeading,
  deviceNoun,
  linkButtonClass,
} from "@/app/onboarding/OnboardingFrame";
import { useNotifications } from "@/app/providers/NotificationProvider";
import { canRevealPaths, revealLogs } from "@/shared/revealPath";

/** Seconds before each automatic retry; the last repeats. */
export const ENGINE_RETRY_DELAYS_S = [2, 4, 8, 15, 30] as const;

/** "Couldn't connect to 127.0.0.1:8080" for a refused request, else the error. */
export function describeEngineError(error: unknown): string {
  if (error instanceof TypeError) {
    let host = api.baseUrl;
    try {
      host = new URL(api.baseUrl, window.location.href).host;
    } catch {
      // Keep the base URL as written.
    }
    return `Couldn't connect to ${host}`;
  }
  return error instanceof Error && error.message ? error.message : String(error);
}

/**
 * The engine can't be reached: retries on its own with backoff, or now on
 * request, until `onRetry` reports success.
 */
export function EngineOffline({
  detail,
  onRetry,
}: {
  detail: string | null;
  /** Resolves true once the engine answers. */
  onRetry: () => Promise<boolean>;
}) {
  const { notify } = useNotifications();
  const attemptRef = useRef(0);
  const mountedRef = useRef(true);
  const [retryAt, setRetryAt] = useState<number | null>(
    () => Date.now() + ENGINE_RETRY_DELAYS_S[0] * 1000,
  );
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const retry = useCallback(async () => {
    setRetryAt(null);
    const ok = await onRetry();
    if (!mountedRef.current || ok) {
      return;
    }
    attemptRef.current += 1;
    const delays = ENGINE_RETRY_DELAYS_S;
    const delay = delays[Math.min(attemptRef.current, delays.length - 1)];
    setNow(Date.now());
    setRetryAt(Date.now() + delay * 1000);
  }, [onRetry]);

  useEffect(() => {
    if (retryAt === null) {
      return;
    }
    const timer = window.setTimeout(() => void retry(), retryAt - Date.now());
    const ticker = window.setInterval(() => setNow(Date.now()), 1000);
    return () => {
      window.clearTimeout(timer);
      window.clearInterval(ticker);
    };
  }, [retry, retryAt]);

  const seconds =
    retryAt === null ? 0 : Math.max(1, Math.ceil((retryAt - now) / 1000));

  return (
    <OnboardingStage>
      <div role="alert">
        <StageHeading
          before={
            <span className="mb-2 flex h-9 w-9 items-center justify-center rounded-full bg-(--danger-bg) text-(--danger-text)">
              <TriangleAlert aria-hidden="true" className="h-[18px] w-[18px]" />
            </span>
          }
          title="Izwi's engine isn't responding"
          description={`Setup needs the engine that runs on this ${deviceNoun()}. It usually starts within a few seconds.`}
        />
      </div>

      <div className="flex flex-col gap-2 border-y border-(--border-muted) py-3.5">
        <span className="flex items-center gap-2 text-[13px] text-(--text-muted)">
          <Loader2 aria-hidden="true" className="h-3.5 w-3.5 animate-spin" />
          {retryAt === null
            ? "Trying again…"
            : `Trying again in ${seconds} second${seconds === 1 ? "" : "s"}`}
        </span>
        {detail ? (
          <code className="wrap-break-word font-(family-name:--app-font-mono) text-[12px] text-(--text-subtle)">
            {detail}
          </code>
        ) : null}
      </div>

      <div className="flex flex-col gap-3.5">
        <PrimaryButton busy={retryAt === null} onClick={() => void retry()}>
          Try again now
        </PrimaryButton>
        {canRevealPaths() ? (
          <p className="text-[13px] text-(--text-muted)">
            Still stuck?{" "}
            <button
              type="button"
              className={linkButtonClass}
              onClick={() =>
                revealLogs().catch((error: unknown) =>
                  notify({
                    title: "Couldn't open the log folder",
                    description: describeEngineError(error),
                    tone: "danger",
                  }),
                )
              }
            >
              Open the log folder
            </button>
          </p>
        ) : null}
      </div>
    </OnboardingStage>
  );
}
