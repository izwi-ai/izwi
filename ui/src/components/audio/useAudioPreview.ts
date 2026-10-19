import { useCallback, useEffect, useRef, useState } from "react";

export interface AudioPreview {
  /** Id of the row whose audio is playing. */
  playingId: string | null;
  /** Play this row's audio, or stop it if it is the one playing. */
  toggle: (id: string, url: string) => void;
  stop: () => void;
}

/**
 * One audio element shared by a list's play buttons: starting a row stops
 * the previous one, and leaving the page stops playback.
 */
export function useAudioPreview(
  onError?: (id: string) => void,
): AudioPreview {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const playingIdRef = useRef<string | null>(null);
  const [playingId, setPlayingIdState] = useState<string | null>(null);
  const onErrorRef = useRef(onError);
  useEffect(() => {
    onErrorRef.current = onError;
  });

  const setPlayingId = (id: string | null) => {
    playingIdRef.current = id;
    setPlayingIdState(id);
  };

  const stop = useCallback(() => {
    audioRef.current?.pause();
    setPlayingId(null);
  }, []);

  const toggle = useCallback((id: string, url: string) => {
    if (playingIdRef.current === id) {
      audioRef.current?.pause();
      setPlayingId(null);
      return;
    }
    let audio = audioRef.current;
    if (!audio) {
      audio = new Audio();
      audio.addEventListener("ended", () => setPlayingId(null));
      audioRef.current = audio;
    }
    audio.pause();
    audio.setAttribute("src", url);
    setPlayingId(id);
    audio.play().catch(() => {
      if (playingIdRef.current === id) {
        setPlayingId(null);
        onErrorRef.current?.(id);
      }
    });
  }, []);

  useEffect(
    () => () => {
      const audio = audioRef.current;
      if (audio) {
        audio.pause();
        audio.removeAttribute("src");
      }
    },
    [],
  );

  return { playingId, toggle, stop };
}
