import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, type StudioProjectRecord } from "@/api";

const SPEEDS = [1, 1.25, 1.5, 0.75] as const;

interface Track {
  segmentId: string;
  url: string;
  duration: number;
  offset: number;
}

/**
 * Plays a project's paragraphs one after another from one audio element,
 * skipping paragraphs without audio, and maps one scrubber over all of
 * them.
 */
export function useProjectPlayer(project: StudioProjectRecord | null) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [segmentId, setSegmentId] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [speed, setSpeed] = useState<number>(1);

  const tracks = useMemo<Track[]>(() => {
    const list: Track[] = [];
    let offset = 0;
    for (const segment of project?.segments ?? []) {
      if (!segment.speech_record_id) continue;
      const duration = segment.audio_duration_secs ?? 0;
      list.push({
        segmentId: segment.id,
        url: api.studioProjectSegmentAudioUrl(
          segment.project_id,
          segment.id,
          segment.speech_record_id,
        ),
        duration,
        offset,
      });
      offset += duration;
    }
    return list;
  }, [project]);
  const tracksRef = useRef(tracks);
  useEffect(() => {
    tracksRef.current = tracks;
  });
  const total = tracks.reduce((sum, track) => sum + track.duration, 0);

  const audio = useCallback(() => {
    if (!audioRef.current) {
      const element = new Audio();
      element.addEventListener("timeupdate", () => setTime(element.currentTime));
      element.addEventListener("pause", () => setPlaying(false));
      element.addEventListener("play", () => setPlaying(true));
      audioRef.current = element;
    }
    return audioRef.current;
  }, []);

  const start = useCallback(
    (track: Track, at = 0) => {
      const element = audio();
      if (element.getAttribute("src") !== track.url) {
        element.setAttribute("src", track.url);
      }
      element.playbackRate = speed;
      element.currentTime = at;
      setSegmentId(track.segmentId);
      setTime(at);
      void element.play().catch(() => setPlaying(false));
    },
    [audio, speed],
  );

  // Move on to the next paragraph with audio when one ends.
  useEffect(() => {
    const element = audio();
    const onEnded = () => {
      const list = tracksRef.current;
      const index = list.findIndex((track) => track.segmentId === segmentId);
      const next = index >= 0 ? list[index + 1] : undefined;
      if (next) {
        start(next);
      } else {
        setSegmentId(null);
        setTime(0);
      }
    };
    element.addEventListener("ended", onEnded);
    return () => element.removeEventListener("ended", onEnded);
  }, [audio, segmentId, start]);

  useEffect(
    () => () => {
      audioRef.current?.pause();
      audioRef.current?.removeAttribute("src");
    },
    [],
  );

  // A different project stops playback.
  const projectId = project?.id ?? null;
  useEffect(() => {
    audioRef.current?.pause();
    setSegmentId(null);
    setTime(0);
  }, [projectId]);

  /** Play from a paragraph, or from the next one after it that has audio. */
  const playFrom = useCallback(
    (fromSegmentId: string) => {
      const order = project?.segments.map((segment) => segment.id) ?? [];
      const position = order.indexOf(fromSegmentId);
      const track = tracksRef.current.find(
        (candidate) => order.indexOf(candidate.segmentId) >= position,
      );
      if (track) start(track);
    },
    [project, start],
  );

  const toggle = useCallback(() => {
    const element = audio();
    if (playing) {
      element.pause();
      return;
    }
    const current = tracksRef.current.find((track) => track.segmentId === segmentId);
    if (current) {
      void element.play().catch(() => setPlaying(false));
    } else if (tracksRef.current[0]) {
      start(tracksRef.current[0]);
    }
  }, [audio, playing, segmentId, start]);

  /** Pause if this paragraph is playing, else play from it. */
  const toggleParagraph = useCallback(
    (paragraphId: string) => {
      if (segmentId === paragraphId && playing) {
        audio().pause();
      } else if (segmentId === paragraphId) {
        void audio().play().catch(() => setPlaying(false));
      } else {
        playFrom(paragraphId);
      }
    },
    [audio, playFrom, playing, segmentId],
  );

  const step = useCallback(
    (direction: 1 | -1) => {
      const list = tracksRef.current;
      const index = list.findIndex((track) => track.segmentId === segmentId);
      const target = list[Math.max(0, index + direction)];
      if (target) start(target);
    },
    [segmentId, start],
  );

  const seek = useCallback(
    (fraction: number) => {
      const at = Math.max(0, Math.min(1, fraction)) * total;
      const list = tracksRef.current;
      const track =
        list.find((candidate) => at < candidate.offset + candidate.duration) ??
        list[list.length - 1];
      if (track) start(track, Math.max(0, at - track.offset));
    },
    [start, total],
  );

  const cycleSpeed = useCallback(() => {
    setSpeed((current) => {
      const next = SPEEDS[(SPEEDS.indexOf(current as (typeof SPEEDS)[number]) + 1) % SPEEDS.length];
      if (audioRef.current) audioRef.current.playbackRate = next;
      return next;
    });
  }, []);

  const currentTrack = tracks.find((track) => track.segmentId === segmentId);
  return {
    segmentId,
    playing,
    /** Seconds into the current paragraph. */
    time,
    /** Seconds into the whole project. */
    position: (currentTrack?.offset ?? 0) + (currentTrack ? time : 0),
    total,
    hasAudio: tracks.length > 0,
    speed,
    playFrom,
    toggle,
    toggleParagraph,
    previous: () => step(-1),
    next: () => step(1),
    seek,
    cycleSpeed,
    /** Paragraph boundaries as fractions of the whole, for the scrubber. */
    marks: total > 0 ? tracks.slice(1).map((track) => track.offset / total) : [],
  };
}

export type ProjectPlayer = ReturnType<typeof useProjectPlayer>;
