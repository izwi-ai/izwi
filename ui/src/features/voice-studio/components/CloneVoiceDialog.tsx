import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type DragEvent,
  type ReactNode,
} from "react";
import { Check, Loader2, Mic, Square, Upload, X } from "lucide-react";

import { api, type ModelInfo } from "@/api";
import { useAudioPreview } from "@/components/audio/useAudioPreview";
import { CreationDialog } from "@/components/CreationDialog";
import { CreationBody, CreationField } from "@/components/creation/CreationParts";
import { formatClockDuration } from "@/components/history/format";
import {
  SegmentedFilter,
  type SegmentOption,
} from "@/components/history/HistoryToolbar";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { AUDIO_FILE_ACCEPT, AUDIO_FILE_HINT } from "@/shared/audioFileAccept";
import { blobToBase64Payload } from "@/utils/audioBase64";
import {
  MAX_REFERENCE_SECS,
  MIN_REFERENCE_SECS,
  prepareReferenceClip,
  type ReferenceClip,
} from "./referenceClip";
import {
  FIELD_LABEL_CLASS,
  FormError,
  PlayButton,
  TEXTAREA_CLASS,
  TEXT_INPUT_CLASS,
  VoiceModelRow,
  VoiceSavedBody,
} from "./NewVoiceParts";

type ClipSource = "upload" | "record";

interface Clip extends ReferenceClip {
  name: string;
}

interface ActiveRecording {
  recorder: MediaRecorder;
  stream: MediaStream;
  chunks: Blob[];
  tick: number;
  limit: number;
}

const SOURCE_OPTIONS: SegmentOption<ClipSource>[] = [
  { id: "upload", label: "Upload" },
  { id: "record", label: "Record" },
];
const NO_CLIPS: Record<ClipSource, Clip | null> = { upload: null, record: null };
const LENGTH_RANGE = `${MIN_REFERENCE_SECS} to ${MAX_REFERENCE_SECS} seconds`;
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

function countWords(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

/** m:ss, rounded away from the 3–30 s range so a clip outside it never reads as inside. */
function formatClipLength(secs: number): string {
  if (secs < MIN_REFERENCE_SECS) {
    return formatClockDuration(Math.floor(secs));
  }
  if (secs > MAX_REFERENCE_SECS) {
    return formatClockDuration(Math.ceil(secs));
  }
  return formatClockDuration(secs);
}

/** The clip is sent as WAV, so its file name says so. */
function wavFilename(name: string): string {
  return `${name.replace(/\.[^./\\]+$/, "") || "reference"}.wav`;
}

/** An object URL for the blob, revoked when the blob changes or goes away. */
function useObjectUrl(blob: Blob | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!blob) {
      setUrl(null);
      return;
    }
    const next = URL.createObjectURL(blob);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [blob]);
  return url;
}

interface CloneVoiceDialogProps {
  open: boolean;
  onClose: () => void;
  /** The reference-capable model that will speak with the voice. */
  model: string | null;
  modelInfo?: ModelInfo | null;
  modelReady: boolean;
  isModelManagerOpen?: boolean;
  onOpenModelManager: () => void;
  onSaved: (voiceId: string) => void;
  onUseInTextToSpeech?: (voiceId: string) => void;
}

export function CloneVoiceDialog({
  open,
  onClose,
  model,
  modelInfo = null,
  modelReady,
  isModelManagerOpen = false,
  onOpenModelManager,
  onSaved,
  onUseInTextToSpeech,
}: CloneVoiceDialogProps) {
  const [name, setName] = useState("");
  const [source, setSource] = useState<ClipSource>("upload");
  const [clips, setClips] = useState(NO_CLIPS);
  const [readingClip, setReadingClip] = useState(false);
  const [recording, setRecording] = useState<"idle" | "starting" | "recording">(
    "idle",
  );
  const [elapsedSecs, setElapsedSecs] = useState(0);
  const [transcript, setTranscript] = useState("");
  const [consent, setConsent] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<{ id: string; name: string } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const recordingRef = useRef<ActiveRecording | null>(null);
  // Bumped when the dialog closes, so microphone, decode and save results
  // that arrive afterwards are dropped.
  const sessionRef = useRef(0);
  const readRef = useRef(0);
  const preview = useAudioPreview();

  const clip = clips[source];
  const clipUrl = useObjectUrl(clip?.blob ?? null);
  const stopPreview = preview.stop;
  useEffect(() => {
    stopPreview();
  }, [clipUrl, stopPreview]);

  const discardRecording = useCallback(() => {
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

  useEffect(() => {
    if (open) {
      return;
    }
    sessionRef.current += 1;
    readRef.current += 1;
    discardRecording();
    setName("");
    setSource("upload");
    setClips(NO_CLIPS);
    setReadingClip(false);
    setRecording("idle");
    setElapsedSecs(0);
    setTranscript("");
    setConsent(false);
    setDragging(false);
    setSaving(false);
    setError(null);
    setSaved(null);
  }, [discardRecording, open]);

  useEffect(
    () => () => {
      sessionRef.current += 1;
      discardRecording();
    },
    [discardRecording],
  );

  const readClip = async (
    target: ClipSource,
    input: Blob,
    clipName: string,
    maxSecs?: number,
  ) => {
    const read = ++readRef.current;
    setReadingClip(true);
    setError(null);
    try {
      const prepared = await prepareReferenceClip(input, { maxSecs });
      if (read === readRef.current) {
        setClips((current) => ({ ...current, [target]: { ...prepared, name: clipName } }));
      }
    } catch {
      if (read === readRef.current) {
        setError(
          target === "upload"
            ? `Couldn't read this file. Use ${AUDIO_FILE_HINT}.`
            : "Couldn't read the recording. Try recording again.",
        );
      }
    } finally {
      if (read === readRef.current) {
        setReadingClip(false);
      }
    }
  };

  const stopRecording = () => {
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
      discardRecording();
      setRecording("idle");
      setError("Couldn't finish the recording. Try again.");
    }
  };

  const startRecording = async () => {
    const session = sessionRef.current;
    setError(null);
    setRecording("starting");
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      if (session === sessionRef.current) {
        setRecording("idle");
        setError(MICROPHONE_ERROR);
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
        setRecording("idle");
        const blob = new Blob(active.chunks, { type: recorder.mimeType || "audio/webm" });
        void readClip("record", blob, "Microphone recording", MAX_REFERENCE_SECS);
      };
      recorder.start();
      recordingRef.current = active;
      const startedAt = Date.now();
      active.tick = window.setInterval(() => {
        setElapsedSecs(Math.floor((Date.now() - startedAt) / 1000));
      }, 250);
      // Saved voices are limited to 30 s, so recording ends there.
      active.limit = window.setTimeout(stopRecording, MAX_REFERENCE_SECS * 1000);
      setElapsedSecs(0);
      setRecording("recording");
    } catch {
      stopTracks(stream);
      setRecording("idle");
      setError(MICROPHONE_ERROR);
    }
  };

  const handleDrop = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) {
      void readClip("upload", file, file.name);
    }
  };

  const duration = clip?.durationSecs ?? null;
  const lengthProblem =
    duration === null
      ? null
      : duration < MIN_REFERENCE_SECS
        ? "Too short"
        : duration > MAX_REFERENCE_SECS
          ? "Too long"
          : null;
  const wordCount = countWords(transcript);
  // Chatterbox clones from the audio alone; every other model conditions on
  // what the reference says.
  const transcriptRequired =
    modelInfo?.speech_capabilities?.reference_transcript_required ?? true;
  const busy = recording !== "idle" || readingClip || saving;
  const canSave =
    Boolean(name.trim() && clip) &&
    !lengthProblem &&
    (wordCount > 0 || !transcriptRequired) &&
    consent &&
    !busy;

  const save = async () => {
    if (!canSave || !clip) {
      return;
    }
    const session = sessionRef.current;
    preview.stop();
    setSaving(true);
    setError(null);
    try {
      const voice = await api.createSavedVoice({
        name: name.trim(),
        reference_text: transcript.trim() || undefined,
        audio_base64: await blobToBase64Payload(clip.blob),
        audio_mime_type: "audio/wav",
        audio_filename: wavFilename(clip.name),
        source_route_kind: "voice_cloning",
        consent: true,
      });
      onSaved(voice.id);
      if (session === sessionRef.current) {
        setSaved({ id: voice.id, name: voice.name || name.trim() });
      }
    } catch (err) {
      if (session === sessionRef.current) {
        setError(err instanceof Error ? err.message : "Couldn't save the voice.");
      }
    } finally {
      if (session === sessionRef.current) {
        setSaving(false);
      }
    }
  };

  if (saved) {
    return (
      <CreationDialog
        open={open}
        onClose={onClose}
        className="max-w-lg"
        title="Voice saved"
        onSubmit={
          onUseInTextToSpeech
            ? () => {
                onUseInTextToSpeech(saved.id);
                onClose();
              }
            : undefined
        }
        submitLabel="Use in Text to Speech"
      >
        <VoiceSavedBody name={saved.name} />
      </CreationDialog>
    );
  }

  let recordingArea: ReactNode;
  if (readingClip) {
    recordingArea = (
      <div className="flex items-center gap-3 rounded-[12px] border border-(--border-subtle) bg-(--surface-panel) px-3 py-3.5 text-[13px] text-(--text-muted)">
        <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />
        Reading audio…
      </div>
    );
  } else if (clip) {
    recordingArea = (
      <div className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3 rounded-[12px] border border-(--border-subtle) bg-(--surface-panel) px-3 py-2.5">
        <PlayButton
          playing={preview.playingId === "clip"}
          label="recording"
          onClick={() => {
            if (clipUrl) {
              preview.toggle("clip", clipUrl);
            }
          }}
          disabled={!clipUrl}
        />
        <div className="flex min-w-0 flex-col gap-1">
          <div className="flex items-baseline justify-between gap-3">
            <span className="truncate text-[13.5px] font-medium text-(--text-primary)">
              {clip.name}
            </span>
            <span className="whitespace-nowrap text-[12px] tabular-nums text-(--text-subtle)">
              {formatClipLength(clip.durationSecs)}
            </span>
          </div>
          <div aria-hidden="true" className="flex h-5 items-center gap-[2px] overflow-hidden">
            {clip.peaks.map((peak, index) => (
              <span
                key={index}
                className="w-[3px] shrink-0 rounded-full bg-(--text-muted)"
                style={{ height: `${Math.max(3, Math.round(peak * 20))}px` }}
              />
            ))}
          </div>
        </div>
        <button
          type="button"
          aria-label="Remove recording"
          onClick={() => setClips((current) => ({ ...current, [source]: null }))}
          disabled={saving}
          className="flex h-7 w-7 items-center justify-center rounded-md text-(--text-muted) transition-colors hover:bg-(--accent-soft) hover:text-(--text-primary) disabled:opacity-50"
        >
          <X aria-hidden="true" className="h-4 w-4" />
        </button>
      </div>
    );
  } else if (source === "upload") {
    recordingArea = (
      <>
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          disabled={saving}
          className={cn(
            "flex w-full flex-col items-center justify-center gap-1.5 rounded-[12px] border border-dashed px-4 py-5 text-center transition-colors disabled:cursor-not-allowed disabled:opacity-60",
            dragging
              ? "border-(--text-primary) bg-(--surface-raised)"
              : "border-(--border-strong) bg-(--surface-panel) hover:bg-(--surface-raised)",
          )}
        >
          <Upload aria-hidden="true" className="h-4 w-4 text-(--text-primary)" />
          <span className="text-[13.5px] font-medium text-(--text-primary)">
            Drop a recording here or browse
          </span>
          <span className="text-xs text-(--text-subtle)">
            {AUDIO_FILE_HINT} · {LENGTH_RANGE}
          </span>
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept={AUDIO_FILE_ACCEPT}
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = "";
            if (file) {
              void readClip("upload", file, file.name);
            }
          }}
        />
      </>
    );
  } else {
    const isRecording = recording === "recording";
    recordingArea = (
      <div className="flex items-center gap-3 rounded-[12px] border border-(--border-subtle) bg-(--surface-panel) px-3 py-2.5">
        {isRecording ? (
          <button
            type="button"
            onClick={stopRecording}
            aria-label="Stop recording"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-(--danger-bg) text-(--danger-text)"
          >
            <Square aria-hidden="true" className="h-3 w-3 fill-current" />
          </button>
        ) : (
          <button
            type="button"
            onClick={() => void startRecording()}
            disabled={recording === "starting" || saving}
            aria-label="Start recording"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-(--accent-soft) text-(--text-primary) transition-colors hover:bg-(--surface-raised) disabled:opacity-60"
          >
            {recording === "starting" ? (
              <Loader2 aria-hidden="true" className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Mic aria-hidden="true" className="h-3.5 w-3.5" />
            )}
          </button>
        )}
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <div className="flex items-baseline justify-between gap-3">
            <span className="text-[13.5px] font-medium text-(--text-primary)">
              {isRecording ? "Recording…" : "Record from your microphone"}
            </span>
            {isRecording ? (
              <span className="whitespace-nowrap text-[12px] tabular-nums text-(--text-subtle)">
                {formatClockDuration(elapsedSecs)} / {formatClockDuration(MAX_REFERENCE_SECS)}
              </span>
            ) : null}
          </div>
          {isRecording ? (
            <div aria-hidden="true" className="h-1 overflow-hidden rounded-full bg-(--border-subtle)">
              <div
                className="h-full rounded-full bg-(--danger-text) transition-[width]"
                style={{ width: `${Math.min(100, (elapsedSecs / MAX_REFERENCE_SECS) * 100)}%` }}
              />
            </div>
          ) : (
            <span className="text-xs text-(--text-subtle)">
              Read a few natural sentences. Recording stops at{" "}
              {formatClockDuration(MAX_REFERENCE_SECS)}.
            </span>
          )}
        </div>
      </div>
    );
  }

  return (
    <CreationDialog
      open={open}
      onClose={onClose}
      modal={!isModelManagerOpen}
      blockDismiss={isModelManagerOpen || saving}
      className="max-w-208"
      title="Clone a voice"
      onSubmit={() => void save()}
      submitLabel="Save voice"
      submitDisabled={!canSave}
      submitting={saving}
      footerStart="Saved voices work in Text to Speech and Studio."
    >
      <CreationBody
        main={
          <>
            {/* No autoFocus: the dialog focuses its first field itself, and an
                autoFocus field stops it returning focus to "New voice" on close. */}
            <CreationField label="Name" htmlFor="clone-voice-name">
              <Input
                id="clone-voice-name"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="e.g. Sam, narrator"
                disabled={saving}
                className={TEXT_INPUT_CLASS}
              />
            </CreationField>

            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between gap-3">
                <span className={FIELD_LABEL_CLASS}>Recording</span>
                <SegmentedFilter
                  label="Recording source"
                  options={SOURCE_OPTIONS}
                  value={source}
                  onChange={(next) => {
                    setError(null);
                    setSource(next);
                  }}
                  disabled={busy}
                />
              </div>
              {recordingArea}
            </div>

            <CreationField
              label={
                transcriptRequired
                  ? "What's said in the recording"
                  : "What's said in the recording (optional)"
              }
              htmlFor="clone-voice-transcript"
              hint={
                transcriptRequired
                  ? "Type it exactly as spoken."
                  : "This model clones from the audio alone. Add the words to reuse the voice with models that need them."
              }
            >
              <textarea
                id="clone-voice-transcript"
                value={transcript}
                onChange={(event) => setTranscript(event.target.value)}
                rows={3}
                disabled={saving}
                placeholder="Every word the speaker says, in order…"
                className={TEXTAREA_CLASS}
              />
            </CreationField>

            <label
              htmlFor="clone-voice-consent"
              className="flex cursor-pointer items-start gap-3 border-t border-(--border-subtle) pt-4"
            >
              <input
                id="clone-voice-consent"
                type="checkbox"
                checked={consent}
                onChange={(event) => setConsent(event.target.checked)}
                disabled={saving}
                aria-labelledby="clone-voice-consent-title"
                aria-describedby="clone-voice-consent-detail"
                className="app-checkbox mt-0.5 h-4 w-4 shrink-0"
              />
              <span className="flex flex-col">
                <span
                  id="clone-voice-consent-title"
                  className="text-[13.5px] font-medium text-(--text-primary)"
                >
                  I have permission to clone this voice
                </span>
                <span
                  id="clone-voice-consent-detail"
                  className="text-xs leading-4 text-(--text-subtle)"
                >
                  It's my voice, or its owner agreed.
                </span>
              </span>
            </label>
          </>
        }
        side={
          <>
            <VoiceModelRow
              model={model}
              modelInfo={modelInfo}
              ready={modelReady}
              onChange={onOpenModelManager}
              disabled={saving}
            />

            <div className="flex flex-col gap-1.5">
              <span className={FIELD_LABEL_CLASS}>Sample</span>
              <ul className="border-t border-(--border-subtle)">
                <SampleCheck
                  title="Length"
                  detail={lengthProblem ? `${lengthProblem}: ${LENGTH_RANGE}` : LENGTH_RANGE}
                  value={duration === null ? "—" : formatClipLength(duration)}
                  state={duration === null ? "empty" : lengthProblem ? "problem" : "ok"}
                />
                <SampleCheck
                  title="Transcript"
                  detail={transcriptRequired ? "Matches what's said" : "Optional for this model"}
                  value={
                    wordCount === 0
                      ? "—"
                      : `${wordCount} ${wordCount === 1 ? "word" : "words"}`
                  }
                  state={wordCount === 0 && transcriptRequired ? "empty" : "ok"}
                />
              </ul>
            </div>

            <FormError message={error} />
          </>
        }
      />
    </CreationDialog>
  );
}

function SampleCheck({
  title,
  detail,
  value,
  state,
}: {
  title: string;
  detail: string;
  value: string;
  state: "empty" | "ok" | "problem";
}) {
  return (
    <li className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 border-b border-(--border-subtle) py-2.5">
      <span className="flex flex-col">
        <span className="text-[13px] text-(--text-primary)">{title}</span>
        <span
          className={cn(
            "text-[11.5px]",
            state === "problem"
              ? "text-(--danger-text)"
              : "text-(--text-subtle)",
          )}
        >
          {detail}
        </span>
      </span>
      <span
        className={cn(
          "flex items-center gap-1.5 whitespace-nowrap text-[13px] tabular-nums",
          state === "ok"
            ? "text-(--status-positive-text)"
            : state === "problem"
              ? "text-(--danger-text)"
              : "text-(--text-subtle)",
        )}
      >
        {state === "ok" ? (
          <Check aria-hidden="true" className="h-[14px] w-[14px]" strokeWidth={2.5} />
        ) : state === "problem" ? (
          <X aria-hidden="true" className="h-[14px] w-[14px]" strokeWidth={2.5} />
        ) : null}
        {value}
      </span>
    </li>
  );
}
