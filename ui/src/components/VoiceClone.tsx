import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  Mic,
  Square,
  Check,
  X,
  BookmarkPlus,
  Loader2,
  RefreshCw,
  Library,
  AlertCircle,
  Trash2,
  CheckCircle2,
} from "lucide-react";
import clsx from "clsx";
import { api, type SavedVoiceSummary } from "../api";
import { blobToBase64Payload } from "../utils/audioBase64";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Button } from "./ui/button";

export interface VoiceCloneReferenceState {
  mode: "upload" | "record" | "saved" | null;
  sampleReady: boolean;
  sampleDurationSecs: number | null;
  transcriptChars: number;
  activeSavedVoiceId: string | null;
  warnings: string[];
  canClone: boolean;
}

interface VoiceCloneProps {
  onVoiceCloneReady: (audioBase64: string, transcript: string) => void;
  onClear: () => void;
  onReferenceStateChange?: (state: VoiceCloneReferenceState) => void;
  onSavedVoiceCreated?: (voiceId: string) => void;
}

function downmixToMono(audioBuffer: AudioBuffer): Float32Array {
  const frameCount = audioBuffer.length;
  const channelCount = audioBuffer.numberOfChannels;
  const mono = new Float32Array(frameCount);

  if (channelCount === 1) {
    mono.set(audioBuffer.getChannelData(0));
    return mono;
  }

  for (let channel = 0; channel < channelCount; channel += 1) {
    const data = audioBuffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i += 1) {
      mono[i] += data[i] / channelCount;
    }
  }

  return mono;
}

function encodeWavPcm16(
  samples: Float32Array,
  sampleRate: number,
): ArrayBuffer {
  const bytesPerSample = 2;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    const intSample =
      sample < 0 ? Math.round(sample * 0x8000) : Math.round(sample * 0x7fff);
    view.setInt16(offset, intSample, true);
    offset += bytesPerSample;
  }

  return buffer;
}

async function normalizeToWavBlob(
  inputBlob: Blob,
): Promise<{ blob: Blob; durationSecs: number }> {
  const arrayBuffer = await inputBlob.arrayBuffer();
  const audioContext = new AudioContext();

  try {
    const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const monoSamples = downmixToMono(decoded);
    const wavBuffer = encodeWavPcm16(monoSamples, decoded.sampleRate);
    return {
      blob: new Blob([wavBuffer], { type: "audio/wav" }),
      durationSecs: decoded.duration,
    };
  } finally {
    void audioContext.close();
  }
}

export function VoiceClone({
  onVoiceCloneReady,
  onClear,
  onReferenceStateChange,
  onSavedVoiceCreated,
}: VoiceCloneProps) {
  const [mode, setMode] = useState<"upload" | "record" | "saved" | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioDurationSecs, setAudioDurationSecs] = useState<number | null>(null);
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isConfirmed, setIsConfirmed] = useState(false);
  const [savedVoices, setSavedVoices] = useState<SavedVoiceSummary[]>([]);
  const [savedVoicesLoading, setSavedVoicesLoading] = useState(false);
  const [savedVoicesError, setSavedVoicesError] = useState<string | null>(null);
  const [selectedSavedVoiceId, setSelectedSavedVoiceId] = useState("");
  const [isApplyingSavedVoice, setIsApplyingSavedVoice] = useState(false);
  const [saveVoiceName, setSaveVoiceName] = useState("");
  const [isSavingVoice, setIsSavingVoice] = useState(false);
  const [saveVoiceStatus, setSaveVoiceStatus] = useState<{
    tone: "success" | "error";
    message: string;
  } | null>(null);
  const [isCreateVoiceModalOpen, setIsCreateVoiceModalOpen] = useState(false);
  const [createVoiceName, setCreateVoiceName] = useState("");
  const [createVoiceTranscript, setCreateVoiceTranscript] = useState("");
  const [createVoiceAudioBlob, setCreateVoiceAudioBlob] = useState<Blob | null>(
    null,
  );
  const [createVoiceAudioUrl, setCreateVoiceAudioUrl] = useState<string | null>(
    null,
  );
  const [createVoiceInputMode, setCreateVoiceInputMode] = useState<
    "upload" | "record" | null
  >(null);
  const [createVoiceIsRecording, setCreateVoiceIsRecording] = useState(false);
  const [createVoiceSaving, setCreateVoiceSaving] = useState(false);
  const [createVoiceError, setCreateVoiceError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const createVoiceFileInputRef = useRef<HTMLInputElement>(null);
  const createVoiceRecorderRef = useRef<MediaRecorder | null>(null);
  const createVoiceChunksRef = useRef<Blob[]>([]);
  const isConfirmingRef = useRef(false);

  const loadSavedVoices = useCallback(async () => {
    setSavedVoicesLoading(true);
    setSavedVoicesError(null);
    try {
      const records = await api.listSavedVoices();
      setSavedVoices(records);
    } catch (err) {
      setSavedVoicesError(
        err instanceof Error ? err.message : "Failed to load saved voices.",
      );
    } finally {
      setSavedVoicesLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadSavedVoices();
  }, [loadSavedVoices]);

  useEffect(() => {
    return () => {
      if (createVoiceAudioUrl) {
        URL.revokeObjectURL(createVoiceAudioUrl);
      }
    };
  }, [createVoiceAudioUrl]);

  const prepareAudioBlob = useCallback(
    async (inputBlob: Blob, inputMode: "upload" | "record" | "saved") => {
      try {
        const normalizedAudio = await normalizeToWavBlob(inputBlob);
        if (audioUrl) {
          URL.revokeObjectURL(audioUrl);
        }
        setAudioBlob(normalizedAudio.blob);
        setAudioUrl(URL.createObjectURL(normalizedAudio.blob));
        setAudioDurationSecs(normalizedAudio.durationSecs);
        setMode(inputMode);
        setError(null);
        setIsConfirmed(false);
      } catch (err) {
        console.error("[VoiceClone] Failed to normalize audio to WAV:", err);
        setError(
          "Could not process this audio format. Please upload/record a standard audio file.",
        );
      }
    },
    [audioUrl],
  );

  const resetCreateVoiceDraft = useCallback(() => {
    if (createVoiceAudioUrl) {
      URL.revokeObjectURL(createVoiceAudioUrl);
    }
    setCreateVoiceName("");
    setCreateVoiceTranscript("");
    setCreateVoiceAudioBlob(null);
    setCreateVoiceAudioUrl(null);
    setCreateVoiceInputMode(null);
      setCreateVoiceError(null);
      setCreateVoiceSaving(false);
      setCreateVoiceIsRecording(false);
  }, [createVoiceAudioUrl]);

  const prepareCreateVoiceAudioBlob = useCallback(
    async (inputBlob: Blob) => {
      try {
        const normalizedAudio = await normalizeToWavBlob(inputBlob);
        if (createVoiceAudioUrl) {
          URL.revokeObjectURL(createVoiceAudioUrl);
        }
        setCreateVoiceAudioBlob(normalizedAudio.blob);
        setCreateVoiceAudioUrl(URL.createObjectURL(normalizedAudio.blob));
        setCreateVoiceError(null);
      } catch (err) {
        console.error(
          "[VoiceClone] Failed to normalize modal voice audio:",
          err,
        );
        setCreateVoiceError(
          "Could not process this recording. Please try recording again.",
        );
      }
    },
    [createVoiceAudioUrl],
  );

  const openCreateVoiceModal = useCallback(() => {
    resetCreateVoiceDraft();
    setCreateVoiceTranscript(transcript.trim());
    setIsCreateVoiceModalOpen(true);
  }, [resetCreateVoiceDraft, transcript]);

  const closeCreateVoiceModal = useCallback(() => {
    if (createVoiceSaving) {
      return;
    }
    if (createVoiceRecorderRef.current && createVoiceIsRecording) {
      createVoiceRecorderRef.current.stop();
    }
    setIsCreateVoiceModalOpen(false);
    resetCreateVoiceDraft();
  }, [createVoiceIsRecording, createVoiceSaving, resetCreateVoiceDraft]);

  useEffect(() => {
    if (!isCreateVoiceModalOpen) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeCreateVoiceModal();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [closeCreateVoiceModal, isCreateVoiceModalOpen]);

  const qualityWarnings = useMemo(() => {
    const warnings: string[] = [];
    const transcriptChars = transcript.trim().length;

    if (audioDurationSecs !== null && audioDurationSecs < 3) {
      warnings.push("Reference sample is short. Aim for at least 3 seconds of clean speech.");
    }
    if (audioDurationSecs !== null && audioDurationSecs > 45) {
      warnings.push("Reference sample is long. Trim it to the cleanest 30-45 seconds.");
    }
    if (transcriptChars > 0 && transcriptChars < 20) {
      warnings.push("Transcript is short. Include the full spoken line for better matching.");
    }
    if (audioBlob && transcriptChars === 0) {
      warnings.push("Add the transcript for the spoken sample before cloning.");
    }

    return warnings;
  }, [audioBlob, audioDurationSecs, transcript]);

  useEffect(() => {
    onReferenceStateChange?.({
      mode,
      sampleReady: Boolean(audioBlob && transcript.trim()),
      sampleDurationSecs: audioDurationSecs,
      transcriptChars: transcript.trim().length,
      activeSavedVoiceId: mode === "saved" ? selectedSavedVoiceId || null : null,
      warnings: qualityWarnings,
      canClone: Boolean(audioBlob && transcript.trim()),
    });
  }, [
    audioBlob,
    audioDurationSecs,
    mode,
    onReferenceStateChange,
    qualityWarnings,
    selectedSavedVoiceId,
    transcript,
  ]);

  // Auto-confirm voice cloning when both audio and transcript are available
  const autoConfirm = useCallback(() => {
    if (
      !audioBlob ||
      !transcript.trim() ||
      isConfirmed ||
      isConfirmingRef.current
    ) {
      return;
    }

    isConfirmingRef.current = true;

    const reader = new FileReader();
    reader.onloadend = () => {
      const base64 = reader.result as string;
      const base64Audio = base64.split(",")[1];
      if (base64Audio) {
        console.log(
          "[VoiceClone] Auto-confirming voice clone - audio length:",
          base64Audio.length,
          "transcript:",
          transcript.trim(),
        );
        onVoiceCloneReady(base64Audio, transcript.trim());
        setIsConfirmed(true);
      }
      isConfirmingRef.current = false;
    };
    reader.onerror = () => {
      console.error("[VoiceClone] Auto-confirm FileReader error");
      isConfirmingRef.current = false;
    };
    reader.readAsDataURL(audioBlob);
  }, [audioBlob, transcript, isConfirmed, onVoiceCloneReady]);

  // Trigger auto-confirm when audio becomes available (transcript already exists)
  // or when transcript is entered (audio already exists)
  useEffect(() => {
    if (audioBlob && transcript.trim()) {
      if (!isConfirmed) {
        // Initial auto-confirm with delay to debounce rapid transcript changes
        const timer = setTimeout(autoConfirm, 300);
        return () => clearTimeout(timer);
      } else {
        // Already confirmed - update parent with new transcript (debounced)
        const timer = setTimeout(() => {
          if (!isConfirmingRef.current) {
            isConfirmingRef.current = true;
            const reader = new FileReader();
            reader.onloadend = () => {
              const base64 = reader.result as string;
              const base64Audio = base64.split(",")[1];
              if (base64Audio) {
                console.log(
                  "[VoiceClone] Updating transcript - audio length:",
                  base64Audio.length,
                  "transcript:",
                  transcript.trim(),
                );
                onVoiceCloneReady(base64Audio, transcript.trim());
              }
              isConfirmingRef.current = false;
            };
            reader.onerror = () => {
              isConfirmingRef.current = false;
            };
            reader.readAsDataURL(audioBlob);
          }
        }, 500);
        return () => clearTimeout(timer);
      }
    }
  }, [audioBlob, transcript, isConfirmed, autoConfirm, onVoiceCloneReady]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith("audio/")) {
      setError("Please upload an audio file");
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError("File size must be less than 10MB");
      return;
    }

    setSelectedSavedVoiceId("");
    setSaveVoiceStatus(null);
    void prepareAudioBlob(file, "upload");
  };

  const handleSavedVoiceSelect = async (voiceId: string) => {
    setSelectedSavedVoiceId(voiceId);
    if (!voiceId) {
      return;
    }

    setIsApplyingSavedVoice(true);
    setError(null);
    setSaveVoiceStatus(null);

    try {
      const [voice, audioResponse] = await Promise.all([
        api.getSavedVoice(voiceId),
        fetch(api.savedVoiceAudioUrl(voiceId)),
      ]);

      if (!audioResponse.ok) {
        throw new Error(
          `Failed to load saved voice audio (${audioResponse.status})`,
        );
      }

      setTranscript(voice.reference_text);
      const voiceAudio = await audioResponse.blob();
      await prepareAudioBlob(voiceAudio, "saved");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load selected saved voice.",
      );
    } finally {
      setIsApplyingSavedVoice(false);
    }
  };

  const handleCreateVoiceFileUpload = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    if (!file.type.startsWith("audio/")) {
      setCreateVoiceError("Please upload an audio file.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setCreateVoiceError("File size must be less than 10MB.");
      return;
    }

    if (createVoiceRecorderRef.current && createVoiceIsRecording) {
      createVoiceRecorderRef.current.stop();
      setCreateVoiceIsRecording(false);
    }

    setCreateVoiceInputMode("upload");
    setCreateVoiceError(null);
    void prepareCreateVoiceAudioBlob(file);
  };

  const startCreateVoiceRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeTypes = [
        "audio/wav",
        "audio/ogg",
        "audio/ogg;codecs=opus",
        "audio/webm;codecs=opus",
        "audio/webm",
      ];

      let selectedMimeType = "";
      for (const mimeType of mimeTypes) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
          selectedMimeType = mimeType;
          break;
        }
      }

      const options = selectedMimeType
        ? { mimeType: selectedMimeType }
        : undefined;
      const mediaRecorder = new MediaRecorder(stream, options);
      createVoiceRecorderRef.current = mediaRecorder;
      createVoiceChunksRef.current = [];
      const actualMimeType = mediaRecorder.mimeType || "audio/webm";

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          createVoiceChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(createVoiceChunksRef.current, {
          type: actualMimeType,
        });
        void prepareCreateVoiceAudioBlob(blob);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setCreateVoiceInputMode("record");
      setCreateVoiceIsRecording(true);
      setCreateVoiceError(null);
    } catch (err) {
      setCreateVoiceError(
        "Microphone access denied. Please allow microphone access.",
      );
      console.error("[VoiceClone] Modal recording error:", err);
    }
  };

  const stopCreateVoiceRecording = () => {
    if (createVoiceRecorderRef.current && createVoiceIsRecording) {
      createVoiceRecorderRef.current.stop();
      setCreateVoiceIsRecording(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Try to use a format that's more compatible with backend processing
      // Prefer formats in order: wav, ogg, webm
      const mimeTypes = [
        "audio/wav",
        "audio/ogg",
        "audio/ogg;codecs=opus",
        "audio/webm;codecs=opus",
        "audio/webm",
      ];

      let selectedMimeType = "";
      for (const mimeType of mimeTypes) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
          selectedMimeType = mimeType;
          break;
        }
      }

      const options = selectedMimeType
        ? { mimeType: selectedMimeType }
        : undefined;
      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      const actualMimeType = mediaRecorder.mimeType || "audio/webm";

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: actualMimeType });
        void prepareAudioBlob(blob, "record");
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setMode("record");
      setError(null);
      setSelectedSavedVoiceId("");
      setSaveVoiceStatus(null);
    } catch (err) {
      setError("Microphone access denied. Please allow microphone access.");
      console.error("Recording error:", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleSaveVoice = async () => {
    if (!audioBlob || !transcript.trim() || isSavingVoice) {
      return;
    }

    const trimmedName = saveVoiceName.trim();
    if (!trimmedName) {
      setSaveVoiceStatus({
        tone: "error",
        message: "Enter a voice name before saving.",
      });
      return;
    }

    setIsSavingVoice(true);
    setSaveVoiceStatus(null);

    try {
      const audioBase64 = await blobToBase64Payload(audioBlob);
      const createdVoice = await api.createSavedVoice({
        name: trimmedName,
        reference_text: transcript.trim(),
        audio_base64: audioBase64,
        audio_mime_type: audioBlob.type || "audio/wav",
        audio_filename: `voice-clone-saved-${Date.now()}.wav`,
        source_route_kind: "voice_cloning",
      });

      setSaveVoiceName("");
      setSaveVoiceStatus({
        tone: "success",
        message: `Saved voice profile "${trimmedName}".`,
      });
      onSavedVoiceCreated?.(createdVoice.id);
      await loadSavedVoices();
    } catch (err) {
      setSaveVoiceStatus({
        tone: "error",
        message:
          err instanceof Error ? err.message : "Failed to save voice profile.",
      });
    } finally {
      setIsSavingVoice(false);
    }
  };

  const handleCreateVoiceFromModal = async () => {
    if (
      !createVoiceAudioBlob ||
      !createVoiceTranscript.trim() ||
      createVoiceSaving
    ) {
      return;
    }

    const trimmedName = createVoiceName.trim();
    if (!trimmedName) {
      setCreateVoiceError("Enter a voice name before saving.");
      return;
    }

    setCreateVoiceSaving(true);
    setCreateVoiceError(null);

    try {
      const audioBase64 = await blobToBase64Payload(createVoiceAudioBlob);
      const createdVoice = await api.createSavedVoice({
        name: trimmedName,
        reference_text: createVoiceTranscript.trim(),
        audio_base64: audioBase64,
        audio_mime_type: createVoiceAudioBlob.type || "audio/wav",
        audio_filename: `voice-profile-${Date.now()}.wav`,
        source_route_kind: "voice_cloning",
      });

      await loadSavedVoices();
      onSavedVoiceCreated?.(createdVoice.id);
      setMode("saved");
      setIsCreateVoiceModalOpen(false);
      resetCreateVoiceDraft();
      await handleSavedVoiceSelect(createdVoice.id);
    } catch (err) {
      setCreateVoiceError(
        err instanceof Error ? err.message : "Failed to create voice profile.",
      );
    } finally {
      setCreateVoiceSaving(false);
    }
  };

  const handleConfirm = async () => {
    if (!audioBlob || !transcript.trim()) {
      setError("Please provide both audio and transcript");
      return;
    }

    try {
      // Convert blob to base64
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result as string;
        // Remove data URL prefix
        const base64Audio = base64.split(",")[1];
        console.log(
          "[VoiceClone] Calling onVoiceCloneReady with audio length:",
          base64Audio?.length,
          "transcript:",
          transcript.trim(),
        );
        onVoiceCloneReady(base64Audio, transcript.trim());
        setIsConfirmed(true);
      };
      reader.onerror = () => {
        setError("Failed to read audio file");
        console.error("[VoiceClone] FileReader error");
      };
      reader.readAsDataURL(audioBlob);
    } catch (err) {
      setError("Failed to process audio");
      console.error(err);
    }
  };

  return (
    <div className="space-y-4">
      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="p-3 rounded-lg bg-[var(--danger-bg)] border border-[var(--danger-border)] text-[var(--danger-text)] text-sm flex items-start gap-2"
          >
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
            <p>{error}</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Transcript input - always visible */}
      <div>
        <label className="block text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wide mb-2">
          Transcript
          <span className="text-red-500 ml-1">*</span>
        </label>
        <textarea
          value={transcript}
          onChange={(e) => setTranscript(e.target.value)}
          placeholder="Enter what you will say in the recording..."
          rows={3}
          className="textarea text-sm py-3 leading-relaxed bg-[var(--bg-surface-0)]"
        />
        <p className="text-[11px] font-medium text-[var(--text-muted)] mt-1.5">
          Type transcript text, then upload, record, or choose a saved voice
        </p>
      </div>

      {/* Audio controls */}
      {!audioBlob ? (
        <div className="space-y-2 mt-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {/* Upload button */}
            <button
              onClick={() => {
                setMode("upload");
                fileInputRef.current?.click();
              }}
              className={clsx(
                "flex flex-col items-center gap-2 p-4 rounded-xl border transition-colors min-h-[96px] justify-center group",
                mode === "upload"
                  ? "border-[var(--border-strong)] bg-[var(--bg-surface-1)] shadow-sm"
                  : "border-[var(--border-muted)] bg-[var(--bg-surface-0)] hover:bg-[var(--bg-surface-1)] hover:border-[var(--border-strong)]",
              )}
            >
              <Upload
                className={clsx(
                  "w-6 h-6",
                  mode === "upload"
                    ? "text-[var(--text-primary)]"
                    : "text-[var(--text-muted)] group-hover:text-[var(--text-primary)] transition-colors",
                )}
              />
              <span
                className={clsx(
                  "text-xs font-medium",
                  mode === "upload"
                    ? "text-[var(--text-primary)]"
                    : "text-[var(--text-secondary)] group-hover:text-[var(--text-primary)] transition-colors",
                )}
              >
                Upload Audio
              </span>
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              onChange={handleFileUpload}
              className="hidden"
            />

            {/* Record button */}
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={clsx(
                "flex flex-col items-center gap-2 p-4 rounded-xl border transition-colors min-h-[96px] justify-center group relative overflow-hidden",
                isRecording
                  ? "border-red-500/30 bg-red-500/10 text-red-500 shadow-[0_0_15px_rgba(239,68,68,0.1)]"
                  : mode === "record"
                    ? "border-[var(--border-strong)] bg-[var(--bg-surface-1)] shadow-sm"
                    : "border-[var(--border-muted)] bg-[var(--bg-surface-0)] hover:bg-[var(--bg-surface-1)] hover:border-[var(--border-strong)]",
              )}
            >
              {isRecording ? (
                <>
                  <div className="absolute inset-0 bg-red-500/5 animate-pulse" />
                  <Square className="w-6 h-6 fill-current relative" />
                  <span className="text-xs font-semibold relative">
                    Stop Recording
                  </span>
                </>
              ) : (
                <>
                  <Mic
                    className={clsx(
                      "w-6 h-6",
                      mode === "record"
                        ? "text-[var(--text-primary)]"
                        : "text-[var(--text-muted)] group-hover:text-[var(--text-primary)] transition-colors",
                    )}
                  />
                  <span
                    className={clsx(
                      "text-xs font-medium",
                      mode === "record"
                        ? "text-[var(--text-primary)]"
                        : "text-[var(--text-secondary)] group-hover:text-[var(--text-primary)] transition-colors",
                    )}
                  >
                    Record Voice
                  </span>
                </>
              )}
            </button>

            {/* Saved voice dropdown */}
            <button
              onClick={() => {
                setMode("saved");
                setError(null);
                if (!savedVoices.length && !savedVoicesLoading) {
                  void loadSavedVoices();
                }
              }}
              className={clsx(
                "flex flex-col items-center gap-2 p-4 rounded-xl border transition-colors min-h-[96px] justify-center group",
                mode === "saved"
                  ? "border-[var(--border-strong)] bg-[var(--bg-surface-1)] shadow-sm"
                  : "border-[var(--border-muted)] bg-[var(--bg-surface-0)] hover:bg-[var(--bg-surface-1)] hover:border-[var(--border-strong)]",
              )}
            >
              <Library
                className={clsx(
                  "w-6 h-6",
                  mode === "saved"
                    ? "text-[var(--text-primary)]"
                    : "text-[var(--text-muted)] group-hover:text-[var(--text-primary)] transition-colors",
                )}
              />
              <span
                className={clsx(
                  "text-xs font-medium",
                  mode === "saved"
                    ? "text-[var(--text-primary)]"
                    : "text-[var(--text-secondary)] group-hover:text-[var(--text-primary)] transition-colors",
                )}
              >
                Saved Voice
              </span>
            </button>
          </div>

          <div className="flex justify-end pt-2">
            <button
              onClick={openCreateVoiceModal}
              className="btn btn-secondary text-xs h-9 px-4 gap-1.5 rounded-lg border-[var(--border-muted)]"
            >
              <BookmarkPlus className="w-4 h-4" />
              Save New Voice
            </button>
          </div>

          <AnimatePresence>
            {mode === "saved" && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="p-4 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] space-y-3 overflow-hidden mt-2"
              >
                <div className="flex items-center justify-between gap-3">
                  <label className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wider">
                    Select Saved Voice
                  </label>
                  <button
                    onClick={() => void loadSavedVoices()}
                    disabled={savedVoicesLoading}
                    className="p-1.5 text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-surface-2)] rounded-md transition-colors"
                    title="Refresh voices"
                  >
                    <RefreshCw
                      className={clsx(
                        "w-3.5 h-3.5",
                        savedVoicesLoading && "animate-spin",
                      )}
                    />
                  </button>
                </div>

                <div className="flex gap-2">
                  <Select
                    value={selectedSavedVoiceId}
                    onValueChange={setSelectedSavedVoiceId}
                    disabled={savedVoicesLoading || !savedVoices.length}
                  >
                    <SelectTrigger className="w-full bg-[var(--bg-surface-1)] border-[var(--border-muted)]">
                      <SelectValue
                        placeholder={
                          savedVoicesLoading
                            ? "Loading..."
                            : savedVoices.length === 0
                              ? "No saved voices"
                              : "Choose a voice profile"
                        }
                      />
                    </SelectTrigger>
                    <SelectContent className="bg-[var(--bg-surface-1)] border-[var(--border-strong)]">
                      {savedVoices.map((voice) => (
                        <SelectItem key={voice.id} value={voice.id}>
                          <div className="flex flex-col gap-0.5">
                            <span className="font-medium text-[var(--text-primary)]">
                              {voice.name}
                            </span>
                            {voice.reference_text_preview && (
                              <span className="text-[10px] text-[var(--text-subtle)] truncate max-w-[200px]">
                                {voice.reference_text_preview}
                              </span>
                            )}
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>

                  <button
                    onClick={() => {
                      if (selectedSavedVoiceId) {
                        void handleSavedVoiceSelect(selectedSavedVoiceId);
                      }
                    }}
                    disabled={
                      !selectedSavedVoiceId ||
                      isApplyingSavedVoice ||
                      savedVoicesLoading
                    }
                    className="btn btn-secondary shrink-0 px-4 h-9 border-[var(--border-muted)]"
                  >
                    {isApplyingSavedVoice ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      "Apply"
                    )}
                  </button>
                </div>

                <AnimatePresence>
                  {savedVoicesError && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="p-2.5 mt-2 rounded-lg bg-[var(--danger-bg)] border border-[var(--danger-border)] text-[var(--danger-text)] text-xs flex items-center gap-2"
                    >
                      <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                      {savedVoicesError}
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      ) : (
        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-4 sm:p-5">
          <div className="flex items-center justify-between gap-4 mb-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-[var(--bg-surface-2)] text-[var(--text-primary)] border border-[var(--border-muted)] shadow-sm">
                <Check className="w-5 h-5 text-green-500" />
              </div>
              <div>
                <h4 className="text-sm font-semibold text-[var(--text-primary)]">
                  Audio Sample Ready
                </h4>
                <p className="text-xs text-[var(--text-muted)] font-medium mt-0.5">
                  {(audioBlob.size / 1024).toFixed(1)} KB
                </p>
              </div>
            </div>
            <button
              onClick={() => {
                if (audioUrl) {
                  URL.revokeObjectURL(audioUrl);
                }
                setAudioBlob(null);
                setAudioUrl(null);
                setAudioDurationSecs(null);
                setIsConfirmed(false);
                setError(null);
                onClear();
              }}
              className="p-2 rounded-lg text-[var(--text-muted)] hover:bg-[var(--danger-bg)] hover:text-[var(--danger-text)] transition-colors"
              title="Remove sample"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>

          <audio
            src={audioUrl ?? undefined}
            controls
            className="w-full h-11 mb-5"
          />

          {!isConfirmed ? (
            <button
              onClick={handleConfirm}
              className="w-full btn btn-primary h-10 gap-2 font-semibold"
            >
              <Check className="w-4 h-4" />
              Confirm Sample
            </button>
          ) : (
            <div className="space-y-4 pt-4 border-t border-[var(--border-muted)]">
              <div className="flex items-center justify-between">
                <h5 className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wider">
                  Save Custom Voice
                </h5>
              </div>
              <div className="flex gap-2">
                <input
                  value={saveVoiceName}
                  onChange={(event) => setSaveVoiceName(event.target.value)}
                  placeholder="Voice name"
                  className="input flex-1 h-10 text-sm bg-[var(--bg-surface-1)] border-[var(--border-muted)]"
                  disabled={isSavingVoice}
                />
                <button
                  onClick={() => void handleSaveVoice()}
                  disabled={
                    isSavingVoice || !saveVoiceName.trim() || !transcript.trim()
                  }
                  className="btn btn-secondary shrink-0 h-10 px-4 gap-2 border-[var(--border-muted)]"
                >
                  {isSavingVoice ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <BookmarkPlus className="w-4 h-4" />
                      Save
                    </>
                  )}
                </button>
              </div>
              <p className="text-[11px] font-medium text-[var(--text-subtle)] leading-relaxed">
                Stores this audio sample and transcript for one-click reuse in
                cloning.
              </p>
              <AnimatePresence>
                {saveVoiceStatus && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className={clsx(
                      "p-3 rounded-lg border text-xs font-medium flex items-center gap-2",
                      saveVoiceStatus.tone === "success"
                        ? "bg-[var(--status-positive-bg)] border-[var(--status-positive-border)] text-[var(--status-positive-text)]"
                        : "bg-[var(--danger-bg)] border-[var(--danger-border)] text-[var(--danger-text)]",
                    )}
                  >
                    {saveVoiceStatus.tone === "success" ? (
                      <CheckCircle2 className="w-3.5 h-3.5" />
                    ) : (
                      <AlertCircle className="w-3.5 h-3.5" />
                    )}
                    {saveVoiceStatus.message}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}
        </div>
      )}

      <AnimatePresence>
        {isCreateVoiceModalOpen && (
          <motion.div
            className="fixed inset-0 z-[70] bg-black/60 p-4 backdrop-blur-sm flex items-center justify-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={closeCreateVoiceModal}
          >
            <motion.div
              initial={{ y: 20, opacity: 0, scale: 0.95 }}
              animate={{ y: 0, opacity: 1, scale: 1 }}
              exit={{ y: 20, opacity: 0, scale: 0.95 }}
              transition={{ type: "spring", duration: 0.5, bounce: 0.3 }}
              onClick={(event) => event.stopPropagation()}
              className="w-full max-w-xl rounded-2xl border border-[var(--border-strong)] bg-[var(--bg-surface-0)] p-5 sm:p-6 shadow-2xl"
            >
              <div className="flex items-center justify-between gap-4 border-b border-[var(--border-muted)] pb-4 mb-5">
                <div>
                  <h3 className="text-base font-semibold text-[var(--text-primary)]">
                    Create Voice Profile
                  </h3>
                  <p className="mt-1 text-xs font-medium text-[var(--text-subtle)]">
                    Upload or record a sample, set transcript text, then save
                    with a name.
                  </p>
                </div>
                <button
                  onClick={closeCreateVoiceModal}
                  className="p-2 rounded-lg text-[var(--text-muted)] hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)] transition-colors"
                  disabled={createVoiceSaving}
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-5">
                <div>
                  <label className="block text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wider mb-2">
                    Voice Name
                    <span className="text-red-500 ml-1">*</span>
                  </label>
                  <input
                    value={createVoiceName}
                    onChange={(event) => setCreateVoiceName(event.target.value)}
                    className="input h-10 text-sm bg-[var(--bg-surface-1)] border-[var(--border-muted)]"
                    placeholder="e.g. Support Agent Voice"
                    disabled={createVoiceSaving}
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wider mb-2">
                    Transcript
                    <span className="text-red-500 ml-1">*</span>
                  </label>
                  <textarea
                    value={createVoiceTranscript}
                    onChange={(event) =>
                      setCreateVoiceTranscript(event.target.value)
                    }
                    rows={3}
                    className="textarea text-sm py-3 leading-relaxed bg-[var(--bg-surface-1)] border-[var(--border-muted)]"
                    placeholder="Enter exactly what is spoken in the recording..."
                    disabled={createVoiceSaving}
                  />
                </div>

                <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
                  <div className="flex items-center justify-between gap-2 mb-3">
                    <span className="text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wider">
                      Voice Sample
                    </span>
                    <span className="text-[10px] font-medium px-2 py-0.5 rounded-md bg-[var(--bg-surface-2)] text-[var(--text-muted)]">
                      5–20s recommended
                    </span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <button
                      onClick={() => {
                        setCreateVoiceInputMode("upload");
                        createVoiceFileInputRef.current?.click();
                      }}
                      className={clsx(
                        "flex items-center justify-center gap-2 rounded-lg border h-11 text-sm font-medium transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background",
                        createVoiceInputMode === "upload"
                          ? "border-[var(--border-strong)] bg-[var(--bg-surface-3)] text-[var(--text-primary)] shadow-sm"
                          : "border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-[var(--text-secondary)] hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-2)]",
                      )}
                      disabled={createVoiceSaving || createVoiceIsRecording}
                    >
                      <Upload className="w-4 h-4" />
                      Upload Recording
                    </button>
                    <input
                      ref={createVoiceFileInputRef}
                      type="file"
                      accept="audio/*"
                      onChange={handleCreateVoiceFileUpload}
                      className="hidden"
                    />

                    <button
                      onClick={
                        createVoiceIsRecording
                          ? stopCreateVoiceRecording
                          : startCreateVoiceRecording
                      }
                      className={clsx(
                        "flex items-center justify-center gap-2 rounded-lg border h-11 text-sm font-medium transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background",
                        createVoiceIsRecording ||
                          createVoiceInputMode === "record"
                          ? "border-red-500/30 bg-red-500/5 text-red-500 shadow-[0_0_15px_rgba(239,68,68,0.1)]"
                          : "border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-[var(--text-secondary)] hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-2)]",
                      )}
                      disabled={createVoiceSaving}
                    >
                      {createVoiceIsRecording ? (
                        <>
                          <div className="relative flex items-center justify-center w-4 h-4">
                            <span className="absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75 animate-ping"></span>
                            <Square className="relative w-3.5 h-3.5 fill-current" />
                          </div>
                          Stop Recording
                        </>
                      ) : (
                        <>
                          <Mic className="w-4 h-4" />
                          Record Voice
                        </>
                      )}
                    </button>
                  </div>

                  {createVoiceAudioUrl ? (
                    <div className="mt-4 space-y-3">
                      <audio
                        src={createVoiceAudioUrl}
                        controls
                        className="w-full h-10"
                      />
                      <div className="flex justify-end">
                        <button
                          onClick={() => {
                            if (createVoiceAudioUrl) {
                              URL.revokeObjectURL(createVoiceAudioUrl);
                            }
                            setCreateVoiceAudioBlob(null);
                            setCreateVoiceAudioUrl(null);
                          }}
                          className="btn btn-ghost text-xs h-8 px-3 text-[var(--danger-text)] hover:bg-[var(--danger-bg)] hover:text-[var(--danger-text)]"
                          disabled={createVoiceSaving || createVoiceIsRecording}
                        >
                          Clear Sample
                        </button>
                      </div>
                    </div>
                  ) : (
                    <p className="mt-3 text-xs font-medium text-[var(--text-subtle)] text-center py-2">
                      Choose upload or record to attach a sample.
                    </p>
                  )}
                </div>

                <AnimatePresence>
                  {createVoiceError && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="p-3 rounded-lg border text-sm bg-[var(--danger-bg)] border-[var(--danger-border)] text-[var(--danger-text)] flex items-start gap-2"
                    >
                      <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                      <p>{createVoiceError}</p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              <div className="mt-6 pt-5 border-t border-[var(--border-muted)] flex items-center justify-end gap-3">
                <Button
                  variant="outline"
                  onClick={closeCreateVoiceModal}
                  disabled={createVoiceSaving}
                  className="h-10"
                >
                  Cancel
                </Button>
                <Button
                  onClick={() => void handleCreateVoiceFromModal()}
                  disabled={
                    createVoiceSaving ||
                    !createVoiceName.trim() ||
                    !createVoiceTranscript.trim() ||
                    !createVoiceAudioBlob
                  }
                  className="h-10 gap-2"
                >
                  {createVoiceSaving ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <BookmarkPlus className="w-4 h-4" />
                      Save Voice Profile
                    </>
                  )}
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
