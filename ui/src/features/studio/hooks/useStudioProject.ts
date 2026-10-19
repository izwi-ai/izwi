import { useCallback, useEffect, useRef, useState } from "react";

import {
  api,
  type StudioProjectPronunciationRecord,
  type StudioProjectRecord,
} from "@/api";

const RENDER_POLL_INTERVAL_MS = 1000;

/** What a poll compares to decide whether anything on screen changed. */
function snapshot(project: StudioProjectRecord): string {
  return [
    project.updated_at,
    project.active_render_job?.id ?? "",
    ...project.segments.map(
      (segment) =>
        `${segment.id}:${segment.render_state}:${segment.speech_record_id ?? ""}:${segment.updated_at}`,
    ),
  ].join("|");
}

function message(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

export type StudioProjectLoad = "loading" | "ready" | "missing" | "error";

/**
 * One Studio project and its pronunciation rules, polled only while the
 * server renders it. Mount it keyed by project id: a response for another
 * project is dropped.
 */
export function useStudioProject(projectId: string, onError: (message: string) => void) {
  const [project, setProject] = useState<StudioProjectRecord | null>(null);
  const [pronunciations, setPronunciations] = useState<StudioProjectPronunciationRecord[]>([]);
  const [load, setLoad] = useState<StudioProjectLoad>("loading");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const onErrorRef = useRef(onError);
  useEffect(() => {
    onErrorRef.current = onError;
  });

  useEffect(() => {
    let current = true;
    setProject(null);
    setLoad("loading");
    setLoadError(null);
    Promise.all([
      api.getStudioProject(projectId),
      api.listStudioProjectPronunciations(projectId),
    ])
      .then(([loaded, rules]) => {
        if (!current) return;
        setProject(loaded);
        setPronunciations(rules);
        setLoad("ready");
      })
      .catch((error: unknown) => {
        if (!current) return;
        const text = message(error, "Couldn't open this project.");
        setLoad(/not found/i.test(text) ? "missing" : "error");
        setLoadError(text);
      });
    return () => {
      current = false;
    };
  }, [projectId, reloadKey]);

  const accept = useCallback((next: StudioProjectRecord) => {
    if (next.id !== projectId) return;
    setProject((previous) =>
      previous && snapshot(previous) === snapshot(next) ? previous : next,
    );
  }, [projectId]);

  const refresh = useCallback(async () => {
    try {
      accept(await api.getStudioProject(projectId));
    } catch {
      // The next poll or action tries again.
    }
  }, [accept, projectId]);

  const activeJobId = project?.active_render_job?.id ?? null;
  useEffect(() => {
    if (!activeJobId) return;
    const timer = window.setInterval(() => void refresh(), RENDER_POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [activeJobId, refresh]);

  /** Run a change that returns the project; errors go to `onError`. */
  const mutate = useCallback(
    async (
      change: () => Promise<StudioProjectRecord>,
      fallback: string,
    ): Promise<StudioProjectRecord | null> => {
      try {
        const next = await change();
        accept(next);
        return next;
      } catch (error) {
        onErrorRef.current(message(error, fallback));
        return null;
      }
    },
    [accept],
  );

  /** Queue paragraphs (default: everything not current) for rendering. */
  const render = useCallback(
    async (segmentIds?: string[]) => {
      try {
        await api.createStudioProjectRenderJob(projectId, { segment_ids: segmentIds });
        await refresh();
        return true;
      } catch (error) {
        onErrorRef.current(message(error, "Couldn't start rendering."));
        return false;
      }
    },
    [projectId, refresh],
  );

  const cancelRender = useCallback(async () => {
    const jobId = project?.active_render_job?.id;
    if (!jobId) return;
    try {
      await api.cancelStudioProjectRenderJob(projectId, jobId);
      await refresh();
    } catch (error) {
      onErrorRef.current(message(error, "Couldn't stop rendering."));
    }
  }, [project?.active_render_job?.id, projectId, refresh]);

  const addPronunciation = useCallback(
    async (source: string, replacement: string) => {
      try {
        const rule = await api.createStudioProjectPronunciation(projectId, {
          source_text: source,
          replacement_text: replacement,
        });
        setPronunciations((rules) => [rule, ...rules]);
        await refresh();
        return true;
      } catch (error) {
        onErrorRef.current(message(error, "Couldn't add the rule."));
        return false;
      }
    },
    [projectId, refresh],
  );

  const removePronunciation = useCallback(
    async (ruleId: string) => {
      try {
        await api.deleteStudioProjectPronunciation(projectId, ruleId);
        setPronunciations((rules) => rules.filter((rule) => rule.id !== ruleId));
        await refresh();
      } catch (error) {
        onErrorRef.current(message(error, "Couldn't remove the rule."));
      }
    },
    [projectId, refresh],
  );

  return {
    project,
    pronunciations,
    load,
    loadError,
    reload: () => setReloadKey((key) => key + 1),
    refresh,
    mutate,
    render,
    cancelRender,
    addPronunciation,
    removePronunciation,
  };
}

export type StudioProjectState = ReturnType<typeof useStudioProject>;
