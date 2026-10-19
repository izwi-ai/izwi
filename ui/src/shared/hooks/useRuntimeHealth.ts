import { useEffect, useState } from "react";

import { api } from "@/api";
import type { HealthResponse } from "@/shared/api/health";

const HEALTH_POLL_INTERVAL_MS = 15_000;

export type RuntimeHealthState =
  | { status: "loading" }
  | { status: "ready"; health: HealthResponse }
  | { status: "error" };

/**
 * Polls /v1/health every 15 seconds while the page is visible. Changing
 * `refreshKey` (e.g. the loaded model count) refreshes at once.
 */
export function useRuntimeHealth(refreshKey?: unknown): RuntimeHealthState {
  const [state, setState] = useState<RuntimeHealthState>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      if (document.visibilityState === "hidden") {
        return;
      }
      api
        .getHealth()
        .then((health) => {
          if (!cancelled) {
            setState({ status: "ready", health });
          }
        })
        .catch(() => {
          if (!cancelled) {
            setState({ status: "error" });
          }
        });
    };

    load();
    const interval = window.setInterval(load, HEALTH_POLL_INTERVAL_MS);
    document.addEventListener("visibilitychange", load);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", load);
    };
  }, [refreshKey]);

  return state;
}
