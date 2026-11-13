import { useCallback, useEffect, useRef, useState } from "react";

import {
  commandErrorMessage,
  dictationApi,
  dictationAvailable,
  dictationEvents,
} from "@/features/dictation/api";
import {
  mergeDictationSettings,
  revertPatch,
} from "@/features/dictation/settings";
import {
  isResultStatus,
  type DictationHistoryEntry,
  type DictationResultStatus,
  type DictationSettings,
  type DictationSettingsPatch,
  type DictationSnapshot,
  type DictationStatus,
} from "@/features/dictation/types";

/** How long the last result stays on the status line after dictation idles. */
export const RESULT_LINGER_MS = 6000;

export type DictationSettingErrors = Partial<
  Record<keyof DictationSettings, string>
>;

export interface UseDictationResult {
  /** False in the web UI, where none of this exists. */
  available: boolean;
  snapshot: DictationSnapshot | null;
  /** Why the first snapshot could not be read. */
  loadError: string | null;
  /** How the last dictation ended, for a few seconds after it did. */
  lastResult: DictationResultStatus | null;
  /** Live words of the dictation in progress, if the model sends any. */
  partial: string;
  /** Why a setting's last change was refused, by setting. */
  errors: DictationSettingErrors;
  history: DictationHistoryEntry[];
  /** Saves a change; a refused one is put back and reported in `errors`. */
  update: (patch: DictationSettingsPatch) => Promise<boolean>;
  /** Reads the snapshot again (the app also re-checks the speech model). */
  refresh: () => Promise<void>;
  clearHistory: () => Promise<void>;
}

/**
 * The desktop app's dictation state: loads the snapshot, follows its events
 * and saves setting changes.
 */
export function useDictation(): UseDictationResult {
  const available = dictationAvailable();
  const [snapshot, setSnapshot] = useState<DictationSnapshot | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<DictationResultStatus | null>(
    null,
  );
  const [partial, setPartial] = useState("");
  const [errors, setErrors] = useState<DictationSettingErrors>({});
  const [history, setHistory] = useState<DictationHistoryEntry[]>([]);

  const snapshotRef = useRef<DictationSnapshot | null>(null);
  // Saves in flight: until they settle, the settings on screen are the
  // user's own edits, not what a snapshot taken mid-save says.
  const pendingSavesRef = useRef(0);
  const lingerTimerRef = useRef<number | null>(null);

  const commit = useCallback((next: DictationSnapshot) => {
    snapshotRef.current = next;
    setSnapshot(next);
  }, []);

  const applySnapshot = useCallback(
    (next: DictationSnapshot) => {
      const current = snapshotRef.current;
      commit(
        current && pendingSavesRef.current > 0
          ? { ...next, settings: current.settings }
          : next,
      );
    },
    [commit],
  );

  const refreshHistory = useCallback(async () => {
    try {
      const entries = await dictationApi.history();
      setHistory(Array.isArray(entries) ? entries : []);
    } catch {
      // The list keeps what it had.
    }
  }, []);

  const refresh = useCallback(async () => {
    try {
      const next = await dictationApi.snapshot();
      if (next) {
        applySnapshot(next);
        setLoadError(null);
      }
    } catch (error) {
      if (!snapshotRef.current) {
        setLoadError(
          commandErrorMessage(error, "Dictation could not be loaded."),
        );
      }
    }
  }, [applySnapshot]);

  const applyStatus = useCallback(
    (status: DictationStatus) => {
      const current = snapshotRef.current;
      if (current) {
        commit({ ...current, status });
      }
      if (status.state === "idle") {
        // The result that came just before stays up until its timer ends.
        setPartial("");
        return;
      }
      if (lingerTimerRef.current !== null) {
        window.clearTimeout(lingerTimerRef.current);
        lingerTimerRef.current = null;
      }
      if (isResultStatus(status)) {
        setPartial("");
        setLastResult(status);
        lingerTimerRef.current = window.setTimeout(() => {
          lingerTimerRef.current = null;
          setLastResult(null);
        }, RESULT_LINGER_MS);
        if (status.state === "inserted") {
          void refreshHistory();
        }
        return;
      }
      setLastResult(null);
      if (status.state === "starting") {
        setPartial("");
      }
    },
    [commit, refreshHistory],
  );

  useEffect(() => {
    if (!available) {
      return;
    }
    let active = true;
    const unlisteners: Array<() => void> = [];
    const keep = (subscription: Promise<() => void>) => {
      void subscription
        .then((unlisten) => {
          if (active) {
            unlisteners.push(unlisten);
          } else {
            unlisten();
          }
        })
        .catch(() => undefined);
    };

    keep(dictationEvents.status((status) => active && applyStatus(status)));
    keep(dictationEvents.partial((text) => active && setPartial(text)));
    keep(dictationEvents.snapshot((next) => active && applySnapshot(next)));

    void refresh();
    void refreshHistory();

    // Permissions are granted in System Settings, away from this window.
    const handleFocus = () => void refresh();
    window.addEventListener("focus", handleFocus);

    return () => {
      active = false;
      window.removeEventListener("focus", handleFocus);
      for (const unlisten of unlisteners) {
        unlisten();
      }
      if (lingerTimerRef.current !== null) {
        window.clearTimeout(lingerTimerRef.current);
        lingerTimerRef.current = null;
      }
    };
  }, [available, applySnapshot, applyStatus, refresh, refreshHistory]);

  const update = useCallback(
    async (patch: DictationSettingsPatch): Promise<boolean> => {
      const current = snapshotRef.current;
      if (!current) {
        return false;
      }
      const keys = Object.keys(patch) as (keyof DictationSettings)[];
      const previous = current.settings;
      const next = mergeDictationSettings(previous, patch);

      commit({ ...current, settings: next });
      setErrors((existing) => {
        if (!keys.some((key) => key in existing)) {
          return existing;
        }
        const remaining = { ...existing };
        for (const key of keys) {
          delete remaining[key];
        }
        return remaining;
      });

      pendingSavesRef.current += 1;
      try {
        const saved = await dictationApi.updateSettings(next);
        pendingSavesRef.current -= 1;
        if (saved) {
          applySnapshot(saved);
        }
        return true;
      } catch (error) {
        pendingSavesRef.current -= 1;
        const latest = snapshotRef.current;
        if (latest) {
          // Only what this change touched goes back; later edits stay.
          commit({
            ...latest,
            settings: mergeDictationSettings(
              latest.settings,
              revertPatch(previous, patch),
            ),
          });
        }
        const message = commandErrorMessage(
          error,
          "That change could not be saved.",
        );
        setErrors((existing) => {
          const withError = { ...existing };
          for (const key of keys) {
            withError[key] = message;
          }
          return withError;
        });
        return false;
      }
    },
    [applySnapshot, commit],
  );

  const clearHistory = useCallback(async () => {
    await dictationApi.clearHistory();
    setHistory([]);
  }, []);

  return {
    available,
    snapshot,
    loadError,
    lastResult,
    partial,
    errors,
    history,
    update,
    refresh,
    clearHistory,
  };
}
