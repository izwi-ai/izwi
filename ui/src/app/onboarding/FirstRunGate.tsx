import {
  createContext,
  startTransition,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useNavigate } from "react-router";

import { api, type OnboardingStateResponse } from "@/api";
import { setAnalyticsEnabled } from "@/app/analytics/client";
import { EngineOffline, describeEngineError } from "@/app/onboarding/EngineOffline";
import { FirstRunOnboarding } from "@/app/onboarding/FirstRunOnboarding";
import { OnboardingFrame } from "@/app/onboarding/OnboardingFrame";

type GateState =
  | { status: "checking" }
  | { status: "offline"; detail: string | null }
  | { status: "due"; analyticsOptIn: boolean }
  | { status: "done" };

interface FirstRunContextValue {
  /** Resets the saved state and shows first run again, in place of the app. */
  runSetupAgain: () => Promise<void>;
}

const FirstRunContext = createContext<FirstRunContextValue | null>(null);

export function useFirstRun(): FirstRunContextValue {
  const context = useContext(FirstRunContext);
  if (!context) {
    throw new Error("useFirstRun must be used within FirstRunGate");
  }
  return context;
}

/** Retries of a failed "setup is done": 2s, doubling, at most a minute apart. */
const COMPLETE_RETRY_MS = { first: 2_000, max: 60_000 };

/**
 * Shows first run instead of the app while `/v1/onboarding` says it isn't
 * done, so people who finished it (upgrades included) never see it. When the
 * engine can't be reached it says so and keeps trying.
 */
export function FirstRunGate({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  const [state, setState] = useState<GateState>({ status: "checking" });
  // A reset or unmount bumps the generation, so a save still in flight
  // can't schedule another try.
  const completionRef = useRef({
    started: false,
    generation: 0,
    timer: null as number | null,
  });

  const applyState = useCallback((onboarding: OnboardingStateResponse) => {
    if (onboarding.completed) {
      setState({ status: "done" });
      return;
    }
    setAnalyticsEnabled(onboarding.analytics_opt_in);
    setState({ status: "due", analyticsOptIn: onboarding.analytics_opt_in });
  }, []);

  const check = useCallback(async (): Promise<boolean> => {
    try {
      applyState(await api.getOnboardingState());
      return true;
    } catch (error) {
      setState({ status: "offline", detail: describeEngineError(error) });
      return false;
    }
  }, [applyState]);

  useEffect(() => {
    void check();
  }, [check]);

  const stopCompletionRetries = useCallback(() => {
    const completion = completionRef.current;
    completion.generation += 1;
    if (completion.timer !== null) {
      window.clearTimeout(completion.timer);
      completion.timer = null;
    }
  }, []);

  useEffect(() => stopCompletionRetries, [stopCompletionRetries]);

  // Once, on finish or skip. A failed save retries in the background for as
  // long as the app is open; if it never lands, first run shows next launch.
  const complete = useCallback(() => {
    const completion = completionRef.current;
    if (completion.started) {
      return;
    }
    completion.started = true;
    const generation = completion.generation;
    let delay = COMPLETE_RETRY_MS.first;
    const attempt = () => {
      api.completeOnboarding().catch((error: unknown) => {
        if (completion.generation !== generation) {
          return;
        }
        console.warn("Couldn't save that setup is done; retrying.", error);
        completion.timer = window.setTimeout(attempt, delay);
        delay = Math.min(delay * 2, COMPLETE_RETRY_MS.max);
      });
    };
    attempt();
  }, []);

  const runSetupAgain = useCallback(async () => {
    const onboarding = await api.resetOnboarding();
    stopCompletionRetries();
    completionRef.current.started = false;
    applyState(onboarding);
  }, [applyState, stopCompletionRetries]);

  // The router applies navigation in a transition. Showing the app in the
  // same transition renders it at the new location, not at "/" (whose
  // redirect to /voice would replace that navigation).
  const exit = useCallback(
    (path?: string) => {
      startTransition(() => {
        if (path) {
          navigate(path);
        }
        setState({ status: "done" });
      });
    },
    [navigate],
  );

  const context = useMemo(() => ({ runSetupAgain }), [runSetupAgain]);

  let content: ReactNode;
  switch (state.status) {
    case "checking":
      content = (
        <div className="h-dvh bg-[var(--surface-canvas)]">
          <span role="status" className="sr-only">
            Starting Izwi…
          </span>
        </div>
      );
      break;
    case "offline":
      // Skipping here only opens the app: nothing is saved while the engine
      // is down, so first run is checked again next launch.
      content = (
        <OnboardingFrame onSkip={() => exit()}>
          <EngineOffline detail={state.detail} onRetry={check} />
        </OnboardingFrame>
      );
      break;
    case "due":
      content = (
        <FirstRunOnboarding
          initialAnalyticsOptIn={state.analyticsOptIn}
          onComplete={complete}
          onExit={exit}
        />
      );
      break;
    case "done":
      content = children;
      break;
  }

  return (
    <FirstRunContext.Provider value={context}>{content}</FirstRunContext.Provider>
  );
}
