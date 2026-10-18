import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "@/api";
import { SpeechPcmPlayer } from "@/features/text-to-speech/pcmPlayer";

export interface ReadAloudVoice {
  modelId: string;
  speaker: string;
}

export interface ReadAloudState {
  messageId: string;
  status: "loading" | "playing";
}

interface Session {
  player: SpeechPcmPlayer;
  controller?: AbortController;
  final: boolean;
}

/**
 * Streams one chat reply at a time through a speech model and plays the PCM
 * as it arrives, so long replies start speaking within a second or two.
 */
export function useReadAloud(onError: (message: string) => void) {
  const [state, setState] = useState<ReadAloudState | null>(null);
  const sessionRef = useRef<Session | null>(null);
  const onErrorRef = useRef(onError);
  useEffect(() => {
    onErrorRef.current = onError;
  });

  const stop = useCallback(() => {
    const session = sessionRef.current;
    sessionRef.current = null;
    session?.controller?.abort();
    session?.player.stop();
    setState(null);
  }, []);

  /** Call from the click handler: audio must unlock on the user's gesture. */
  const start = useCallback(
    (messageId: string, text: string, voice: ReadAloudVoice) => {
      stop();
      const player = new SpeechPcmPlayer(new AudioContext(), () => {
        if (sessionRef.current?.player === player) stop();
      });
      const session: Session = { player, final: false };
      const isCurrent = () => sessionRef.current === session;
      const fail = (message: string) => {
        if (!isCurrent()) return;
        stop();
        onErrorRef.current(message);
      };
      sessionRef.current = session;
      setState({ messageId, status: "loading" });

      const unlocked = player.unlock();
      unlocked.catch(() => fail("Audio playback could not start."));
      // Chunks must reach the player in order even though each push awaits.
      let pushes = Promise.resolve();

      session.controller = api.generateTTSStream(
        {
          text,
          model_id: voice.modelId,
          speaker: voice.speaker,
          format: "pcm",
        },
        {
          onStart: (event) => {
            if (!isCurrent()) return;
            try {
              player.start(
                event.requestId,
                event.sampleRate,
                event.audioFormat,
              );
            } catch (error) {
              fail((error as Error).message);
            }
          },
          onChunk: (event) => {
            pushes = pushes.then(async () => {
              if (!isCurrent()) return;
              try {
                await unlocked;
                await player.push(event);
                if (isCurrent()) setState({ messageId, status: "playing" });
              } catch (error) {
                fail((error as Error).message);
              }
            });
          },
          onFinal: () => {
            if (!isCurrent()) return;
            session.final = true;
            void pushes.then(() => {
              if (isCurrent()) player.finish();
            });
          },
          onError: fail,
          onDone: () => {
            if (isCurrent() && !session.final) {
              fail("The speech stream ended before the reply was read.");
            }
          },
        },
      );
    },
    [stop],
  );

  useEffect(() => stop, [stop]);

  return { state, start, stop };
}
