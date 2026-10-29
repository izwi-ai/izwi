import { useCallback, useEffect, useRef, useState } from "react";

export type MicrophoneRecordingStatus = "idle" | "starting" | "recording";

interface ActiveRecording {
  recorder: MediaRecorder;
  stream: MediaStream;
  chunks: Blob[];
  tick: number;
  limit: number;
}

const RECORDER_MIME_TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
const MICROPHONE_ERROR =
  "Couldn't use the microphone. Allow microphone access and try again.";

function stopTracks(stream: MediaStream) {
  stream.getTracks().forEach((track) => track.stop());
}

function createRecorder(stream: MediaStream): MediaRecorder {
  const mimeType = RECORDER_MIME_TYPES.find((type) =>
    MediaRecorder.isTypeSupported(type),
  );
  return mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
}

/**
 * Records from the microphone until stopped or `maxSecs` pass, then hands
 * the recording to `onRecorded` in the browser's own format (usually
 * WebM/Opus). Unmounting discards a recording.
 */
export function useMicrophoneRecording({
  maxSecs,
  onRecorded,
  onError,
}: {
  maxSecs: number;
  onRecorded: (recording: Blob) => void;
  onError: (message: string) => void;
}) {
  const [status, setStatus] = useState<MicrophoneRecordingStatus>("idle");
  const [elapsedSecs, setElapsedSecs] = useState(0);
  const recordingRef = useRef<ActiveRecording | null>(null);
  // Bumped on unmount, so a microphone that opens afterwards is closed.
  const sessionRef = useRef(0);
  const onRecordedRef = useRef(onRecorded);
  const onErrorRef = useRef(onError);
  useEffect(() => {
    onRecordedRef.current = onRecorded;
    onErrorRef.current = onError;
  });

  const discard = useCallback(() => {
    const active = recordingRef.current;
    recordingRef.current = null;
    if (!active) {
      return;
    }
    window.clearInterval(active.tick);
    window.clearTimeout(active.limit);
    active.recorder.ondataavailable = null;
    active.recorder.onstop = null;
    try {
      if (active.recorder.state !== "inactive") {
        active.recorder.stop();
      }
    } catch {
      // The tracks below still have to stop.
    }
    stopTracks(active.stream);
  }, []);

  useEffect(
    () => () => {
      sessionRef.current += 1;
      discard();
    },
    [discard],
  );

  const stop = useCallback(() => {
    const active = recordingRef.current;
    // Already stopping: the recorder hands over its audio shortly.
    if (!active || active.recorder.state === "inactive") {
      return;
    }
    window.clearInterval(active.tick);
    window.clearTimeout(active.limit);
    try {
      active.recorder.stop();
    } catch {
      discard();
      setStatus("idle");
      onErrorRef.current("Couldn't finish the recording. Try again.");
    }
  }, [discard]);

  const start = useCallback(async () => {
    const session = sessionRef.current;
    setStatus("starting");
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      if (session === sessionRef.current) {
        setStatus("idle");
        onErrorRef.current(MICROPHONE_ERROR);
      }
      return;
    }
    if (session !== sessionRef.current) {
      stopTracks(stream);
      return;
    }

    try {
      const recorder = createRecorder(stream);
      const active: ActiveRecording = { recorder, stream, chunks: [], tick: 0, limit: 0 };
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          active.chunks.push(event.data);
        }
      };
      recorder.onstop = () => {
        if (recordingRef.current !== active) {
          return;
        }
        recordingRef.current = null;
        stopTracks(stream);
        setStatus("idle");
        onRecordedRef.current(
          new Blob(active.chunks, { type: recorder.mimeType || "audio/webm" }),
        );
      };
      recorder.start();
      recordingRef.current = active;
      const startedAt = Date.now();
      active.tick = window.setInterval(() => {
        setElapsedSecs(Math.floor((Date.now() - startedAt) / 1000));
      }, 250);
      active.limit = window.setTimeout(stop, maxSecs * 1000);
      setElapsedSecs(0);
      setStatus("recording");
    } catch {
      stopTracks(stream);
      setStatus("idle");
      onErrorRef.current(MICROPHONE_ERROR);
    }
  }, [maxSecs, stop]);

  return { status, elapsedSecs, start, stop };
}
