import { useCallback, useEffect, useMemo, useState } from "react";

import { api, type VoiceObservation } from "@/api";
import { VOICE_AGENT_SYSTEM_PROMPT } from "@/features/voice/realtime/support";

/** Standard runs separate ASR, chat and TTS models; Compact one audio model. */
export type VoiceEngineId = "standard" | "compact";

const ENGINE_STORAGE_KEY = "izwi.voice.engine";
const COMPACT_MODEL_STORAGE_KEY = "izwi.voice.compactModel";
const SPEAKER_STORAGE_KEY = "izwi.voice.speaker";

function readStored(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStored(key: string, value: string) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // A convenience only: the default applies next time.
  }
}

export type PromptSaveState = "idle" | "saving" | "saved" | "error";

/**
 * What the user chose for voice conversations: the engine, the Compact
 * engine's model and the voice are remembered on this device; the
 * instructions and memory setting live in the server's voice profile.
 */
export function useVoicePreferences(onError?: (message: string) => void) {
  const [engine, setEngineState] = useState<VoiceEngineId>(() =>
    readStored(ENGINE_STORAGE_KEY) === "compact" ? "compact" : "standard",
  );
  // Null until one is chosen: the Compact engine then runs its default.
  const [compactModel, setCompactModelState] = useState<string | null>(() =>
    readStored(COMPACT_MODEL_STORAGE_KEY),
  );
  const [speaker, setSpeakerState] = useState<string | null>(() =>
    readStored(SPEAKER_STORAGE_KEY),
  );

  const [savedPrompt, setSavedPrompt] = useState(VOICE_AGENT_SYSTEM_PROMPT);
  const [defaultPrompt, setDefaultPrompt] = useState(VOICE_AGENT_SYSTEM_PROMPT);
  const [promptSaveState, setPromptSaveState] =
    useState<PromptSaveState>("idle");
  const [memoryEnabled, setMemoryEnabledState] = useState(true);
  const [isProfileLoading, setIsProfileLoading] = useState(true);

  const [memories, setMemories] = useState<VoiceObservation[] | null>(null);
  const [isMemoryBusy, setIsMemoryBusy] = useState(false);

  const report = useCallback(
    (err: unknown, fallback: string) => {
      onError?.(err instanceof Error ? err.message : fallback);
    },
    [onError],
  );

  useEffect(() => {
    let cancelled = false;
    api
      .getVoiceProfile()
      .then((profile) => {
        if (cancelled) return;
        setDefaultPrompt(
          profile.default_system_prompt || VOICE_AGENT_SYSTEM_PROMPT,
        );
        setSavedPrompt(
          profile.system_prompt ||
            profile.default_system_prompt ||
            VOICE_AGENT_SYSTEM_PROMPT,
        );
        setMemoryEnabledState(profile.observational_memory_enabled ?? true);
      })
      .catch((err) => {
        if (!cancelled) report(err, "Couldn't load voice settings.");
      })
      .finally(() => {
        if (!cancelled) setIsProfileLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [report]);

  const setEngine = useCallback((next: VoiceEngineId) => {
    setEngineState(next);
    writeStored(ENGINE_STORAGE_KEY, next);
  }, []);

  const setCompactModel = useCallback((next: string) => {
    setCompactModelState(next);
    writeStored(COMPACT_MODEL_STORAGE_KEY, next);
  }, []);

  const setSpeaker = useCallback((next: string) => {
    setSpeakerState(next);
    writeStored(SPEAKER_STORAGE_KEY, next);
  }, []);

  /** Save the instructions; an empty text restores the default. */
  const savePrompt = useCallback(
    async (text: string) => {
      const next = text.trim() || defaultPrompt;
      if (next === savedPrompt) {
        return;
      }
      setPromptSaveState("saving");
      try {
        const profile = await api.updateVoiceProfile({ system_prompt: next });
        setSavedPrompt(profile.system_prompt || next);
        setPromptSaveState("saved");
      } catch (err) {
        setPromptSaveState("error");
        report(err, "Couldn't save the instructions.");
      }
    },
    [defaultPrompt, report, savedPrompt],
  );

  const setMemoryEnabled = useCallback(
    async (next: boolean) => {
      setMemoryEnabledState(next);
      try {
        const profile = await api.updateVoiceProfile({
          observational_memory_enabled: next,
        });
        setMemoryEnabledState(profile.observational_memory_enabled ?? next);
      } catch (err) {
        setMemoryEnabledState(!next);
        report(err, "Couldn't change memory.");
      }
    },
    [report],
  );

  const loadMemories = useCallback(async () => {
    try {
      setMemories(await api.listVoiceObservations(100));
    } catch (err) {
      report(err, "Couldn't load memories.");
    }
  }, [report]);

  const forgetMemory = useCallback(
    async (id: string) => {
      setIsMemoryBusy(true);
      try {
        await api.deleteVoiceObservation(id);
        setMemories((current) =>
          current ? current.filter((memory) => memory.id !== id) : current,
        );
      } catch (err) {
        report(err, "Couldn't forget that memory.");
      } finally {
        setIsMemoryBusy(false);
      }
    },
    [report],
  );

  const forgetAllMemories = useCallback(async () => {
    setIsMemoryBusy(true);
    try {
      await api.clearVoiceObservations();
      setMemories([]);
    } catch (err) {
      report(err, "Couldn't forget memories.");
    } finally {
      setIsMemoryBusy(false);
    }
  }, [report]);

  return useMemo(
    () => ({
      engine,
      setEngine,
      compactModel,
      setCompactModel,
      speaker,
      setSpeaker,
      savedPrompt,
      defaultPrompt,
      isPromptCustom: savedPrompt.trim() !== defaultPrompt.trim(),
      promptSaveState,
      savePrompt,
      memoryEnabled,
      setMemoryEnabled,
      isProfileLoading,
      memories,
      isMemoryBusy,
      loadMemories,
      forgetMemory,
      forgetAllMemories,
    }),
    [
      compactModel,
      defaultPrompt,
      engine,
      forgetAllMemories,
      forgetMemory,
      isMemoryBusy,
      isProfileLoading,
      loadMemories,
      memories,
      memoryEnabled,
      promptSaveState,
      savePrompt,
      savedPrompt,
      setCompactModel,
      setEngine,
      setMemoryEnabled,
      setSpeaker,
      speaker,
    ],
  );
}

export type VoicePreferences = ReturnType<typeof useVoicePreferences>;
