import { useCallback, useEffect, useRef, useState } from "react";

interface UseAudioPlaybackOptions {
  /** Restart from the top when this changes too (e.g. the record id). */
  resetKey?: unknown;
  /** Longest known length (transcript end, record duration), for seeking
   * before metadata loads. */
  durationHint?: number;
  loadErrorMessage?: string;
  playErrorMessage?: string;
}

export interface AudioPlayback {
  src: string | null;
  currentTime: number;
  /** Max of the media duration and the hint. */
  duration: number;
  isPlaying: boolean;
  playbackRate: number;
  error: string | null;
  toggle: () => Promise<void>;
  pause: () => void;
  seek: (time: number) => void;
  skip: (deltaSeconds: number) => void;
  setPlaybackRate: (rate: number) => void;
  /** Spread onto the (hidden) <audio> element. */
  audioProps: {
    ref: React.RefObject<HTMLAudioElement>;
    src: string | undefined;
    preload: "metadata";
    onLoadedMetadata: (event: React.SyntheticEvent<HTMLAudioElement>) => void;
    onTimeUpdate: (event: React.SyntheticEvent<HTMLAudioElement>) => void;
    onPlay: () => void;
    onPause: () => void;
    onRateChange: (event: React.SyntheticEvent<HTMLAudioElement>) => void;
    onError: () => void;
  };
}

/** State and controls for one audio element. */
export function useAudioPlayback(
  src: string | null | undefined,
  {
    resetKey,
    durationHint = 0,
    loadErrorMessage = "Unable to load this audio.",
    playErrorMessage = "Unable to start playback.",
  }: UseAudioPlaybackOptions = {},
): AudioPlayback {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [mediaDuration, setMediaDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRateState] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const duration = Math.max(mediaDuration, durationHint, 0);

  useEffect(() => {
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
      audio.playbackRate = 1;
    }
    setCurrentTime(0);
    setMediaDuration(0);
    setIsPlaying(false);
    setPlaybackRateState(1);
    setError(null);
  }, [src, resetKey]);

  const toggle = useCallback(async () => {
    const audio = audioRef.current;
    if (!audio || !src) {
      return;
    }
    try {
      if (audio.paused) {
        await audio.play();
      } else {
        audio.pause();
      }
    } catch {
      setError(playErrorMessage);
    }
  }, [playErrorMessage, src]);

  const pause = useCallback(() => {
    audioRef.current?.pause();
  }, []);

  const seek = useCallback(
    (time: number) => {
      const clamped = Math.max(0, Math.min(time, duration || 0));
      if (audioRef.current) {
        audioRef.current.currentTime = clamped;
      }
      setCurrentTime(clamped);
    },
    [duration],
  );

  const skip = useCallback(
    (deltaSeconds: number) => seek(currentTime + deltaSeconds),
    [currentTime, seek],
  );

  const setPlaybackRate = useCallback((rate: number) => {
    if (audioRef.current) {
      audioRef.current.playbackRate = rate;
    }
    setPlaybackRateState(rate);
  }, []);

  return {
    src: src ?? null,
    currentTime,
    duration,
    isPlaying,
    playbackRate,
    error,
    toggle,
    pause,
    seek,
    skip,
    setPlaybackRate,
    audioProps: {
      ref: audioRef,
      src: src ?? undefined,
      preload: "metadata",
      onLoadedMetadata: (event) => {
        const value = event.currentTarget.duration;
        setMediaDuration(Number.isFinite(value) ? value : 0);
        setError(null);
      },
      onTimeUpdate: (event) => setCurrentTime(event.currentTarget.currentTime),
      onPlay: () => setIsPlaying(true),
      onPause: () => setIsPlaying(false),
      onRateChange: (event) =>
        setPlaybackRateState(event.currentTarget.playbackRate),
      onError: () => setError(loadErrorMessage),
    },
  };
}
