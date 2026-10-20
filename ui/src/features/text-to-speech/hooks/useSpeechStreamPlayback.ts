import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { api, type SpeechHistoryRecord, type SpeechHistoryRecordCreateRequest,
  type SpeechHistoryRecordStreamCallbacks } from "@/api";
import { SpeechPcmPlayer } from "../pcmPlayer";

interface Session {
  player: SpeechPcmPlayer;
  controller?: AbortController;
  recordId?: string;
  originRecordId?: string;
  final: boolean;
  durable: boolean;
  reject: (error: Error) => void;
}

/** Owned by the route: closing the creation dialog must not terminate playback. */
export function useSpeechStreamPlayback(
  recordId: string | undefined,
  callbacks: SpeechHistoryRecordStreamCallbacks,
) {
  const session = useRef<Session | null>(null);
  const callbacksRef = useRef(callbacks);
  useLayoutEffect(() => { callbacksRef.current = callbacks; }, [callbacks]);
  const [status, setStatus] = useState<"idle" | "generating" | "playing">("idle");
  const [progress, setProgress] = useState<{ completedSegments: number; totalSegments: number; processedTextBytes: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const dispose = useCallback((cancelGeneration: boolean, reason = "Speech generation was stopped.") => {
    const current = session.current;
    if (!current) return;
    session.current = null;
    current.controller?.abort();
    current.player.stop();
    current.reject(new Error(reason));
    if (cancelGeneration && !current.final && current.recordId) {
      void api.cancelTextToSpeechRecord(current.recordId).catch(() => {});
    }
  }, []);

  const stop = useCallback((cancelGeneration = true) => {
    dispose(cancelGeneration);
    setStatus("idle");
  }, [dispose]);

  // Leaving the page or opening another record stops playback, not the
  // take: the server finishes it into history. Only Stop cancels it.
  useEffect(() => () => dispose(false), [dispose]);
  useEffect(() => {
    const current = session.current;
    if (current && (current.recordId ?? current.originRecordId) !== recordId) stop(false);
  }, [recordId, stop]);

  const begin = useCallback((request?: SpeechHistoryRecordCreateRequest, existingRecord?: SpeechHistoryRecord) => {
    dispose(!session.current?.durable);
    setProgress(null);
    setError(null);
    setStatus("generating");
    return new Promise<SpeechHistoryRecord>((resolve, reject) => {
      let current: Session;
      const fail = (message: string) => {
        if (session.current !== current) return;
        dispose(!current.durable, message);
        setStatus("idle");
        setError(message);
        callbacksRef.current.onError?.(message);
        reject(new Error(message));
      };
      try {
        const player = new SpeechPcmPlayer(new AudioContext(), () => {
          if (session.current !== current) return;
          // The final PCM can finish before the SSE done event arrives.
          current.player.stop();
          setStatus("idle");
        });
        current = { player, final: false, durable: !!existingRecord, recordId: existingRecord?.id, reject, originRecordId: recordId };
        session.current = current;
        // resume() must run on the original user gesture, not in onStart.
        const unlocked = player.unlock();
        void unlocked.catch(() => fail("Browser audio playback could not start."));
        const events: SpeechHistoryRecordStreamCallbacks = {
          onDurable: () => {
            if (session.current !== current) return;
            current.durable = true;
            callbacksRef.current.onDurable?.();
          },
          onProgress: (value) => {
            if (session.current !== current) return;
            setProgress(value);
            callbacksRef.current.onProgress?.(value);
          },
          onReconnecting: (attempt) => {
            if (session.current !== current) return;
            callbacksRef.current.onReconnecting?.(attempt);
          },
          onCreated: (record) => {
            if (session.current !== current) return;
            current.recordId = record.id;
            resolve(record);
          },
          onStart: (event) => {
            if (session.current !== current) return;
            try {
              player.start(event.requestId, event.sampleRate, event.audioFormat);
              callbacksRef.current.onStart?.(event);
            } catch (err) { fail((err as Error).message); }
          },
          onChunk: async (event) => {
            if (session.current !== current) return;
            try {
              await unlocked;
              if (session.current !== current) return;
              await player.push(event);
              if (session.current === current) setStatus("playing");
            } catch (err) { fail((err as Error).message); }
          },
          onFinal: (event) => {
            if (session.current !== current) return;
            current.final = true;
            current.recordId = event.record.id;
            callbacksRef.current.onFinal?.(event);
            resolve(event.record);
            player.finish();
          },
          onError: fail,
          onDone: () => {
            if (session.current !== current) return;
            if (!current.final) fail("Speech stream ended before generation completed.");
            else callbacksRef.current.onDone?.();
          },
        };
        current.controller = existingRecord
          ? api.attachTextToSpeechRecordStream(existingRecord.id, events)
          : api.createTextToSpeechRecordStream(request!, events);
        // Also handle synchronous transport callbacks used by adapters and tests.
        if (session.current !== current) current.controller?.abort();
      } catch (err) {
        dispose(!session.current?.durable);
        setStatus("idle");
        reject(err);
      }
    });
  }, [dispose, recordId]);

  const create = useCallback((request: SpeechHistoryRecordCreateRequest) => begin(request), [begin]);
  const listen = useCallback((record: SpeechHistoryRecord) => begin(undefined, record), [begin]);

  return { create, listen, stop, status, error, progress };
}
