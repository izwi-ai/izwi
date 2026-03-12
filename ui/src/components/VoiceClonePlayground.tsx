import { useState, useRef, useMemo, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Users,
  Square,
  Download,
  RotateCcw,
  Loader2,
  CheckCircle2,
  AlertCircle,
  ArrowRight,
  BadgeCheck,
  Globe,
  ChevronDown,
  Settings2,
} from "lucide-react";
import { api, type SpeechHistoryRecord, type TTSGenerationStats } from "../api";
import {
  VOICE_ROUTE_SECTION_LABEL_CLASS,
  VOICE_ROUTE_WORKSPACE_DESCRIPTION_CLASS,
  VOICE_ROUTE_WORKSPACE_TITLE_CLASS,
} from "@/components/voiceRouteTypography";
import { VoiceClone, type VoiceCloneReferenceState } from "./VoiceClone";
import { LANGUAGES } from "../types";
import clsx from "clsx";
import { GenerationStats } from "./GenerationStats";
import { SpeechHistoryPanel } from "./SpeechHistoryPanel";
import { useDownloadIndicator } from "../utils/useDownloadIndicator";

interface ModelOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

interface VoiceClonePlaygroundProps {
  selectedModel: string | null;
  selectedModelReady?: boolean;
  modelOptions?: ModelOption[];
  onSelectModel?: (variant: string) => void;
  onOpenModelManager?: () => void;
  onModelRequired: () => void;
  onUseInTts?: (voiceId: string) => void;
  historyActionContainer?: HTMLElement | null;
}

function revokeObjectUrlIfNeeded(url: string | null): void {
  if (url && url.startsWith("blob:")) {
    URL.revokeObjectURL(url);
  }
}

function mapRecordToStats(record: SpeechHistoryRecord): TTSGenerationStats {
  return {
    generation_time_ms: record.generation_time_ms,
    audio_duration_secs: record.audio_duration_secs ?? 0,
    rtf: record.rtf ?? 0,
    tokens_generated: record.tokens_generated ?? 0,
  };
}

const DEFAULT_AUDITION_TEXT =
  "Thanks for calling Izwi. This is a short preview of your cloned voice for your next project.";

export function VoiceClonePlayground({
  selectedModel,
  selectedModelReady = false,
  modelOptions = [],
  onSelectModel,
  onOpenModelManager,
  onModelRequired,
  onUseInTts,
  historyActionContainer = null,
}: VoiceClonePlaygroundProps) {
  const [text, setText] = useState(DEFAULT_AUDITION_TEXT);
  const [language, setLanguage] = useState("Auto");
  const [showLanguageSelect, setShowLanguageSelect] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [generationStats, setGenerationStats] =
    useState<TTSGenerationStats | null>(null);
  const [latestRecord, setLatestRecord] = useState<SpeechHistoryRecord | null>(
    null,
  );
  const [voiceCloneAudio, setVoiceCloneAudio] = useState<string | null>(null);
  const [voiceCloneTranscript, setVoiceCloneTranscript] = useState<
    string | null
  >(null);
  const [isVoiceReady, setIsVoiceReady] = useState(false);
  const [referenceState, setReferenceState] =
    useState<VoiceCloneReferenceState | null>(null);
  const [consentConfirmed, setConsentConfirmed] = useState(false);
  const [savedVoiceId, setSavedVoiceId] = useState<string | null>(null);
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);
  const {
    downloadState,
    downloadMessage,
    isDownloading,
    beginDownload,
    completeDownload,
    failDownload,
  } = useDownloadIndicator();

  const audioRef = useRef<HTMLAudioElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);

  const selectedOption = useMemo(() => {
    if (!selectedModel) {
      return null;
    }
    return (
      modelOptions.find((option) => option.value === selectedModel) || null
    );
  }, [selectedModel, modelOptions]);

  useEffect(() => {
    const onPointerDown = (event: MouseEvent) => {
      if (
        modelMenuRef.current &&
        event.target instanceof Node &&
        !modelMenuRef.current.contains(event.target)
      ) {
        setIsModelMenuOpen(false);
      }
    };
    window.addEventListener("mousedown", onPointerDown);
    return () => window.removeEventListener("mousedown", onPointerDown);
  }, []);

  const handleGenerate = async () => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      return;
    }

    if (!text.trim()) {
      setError("Please enter some text to synthesize");
      return;
    }

    if (!voiceCloneAudio || !voiceCloneTranscript) {
      setError("Please provide a voice reference (audio + transcript)");
      return;
    }

    if (!consentConfirmed) {
      setError(
        "Confirm that you have permission to clone this voice before generating audio.",
      );
      return;
    }

    try {
      setGenerating(true);
      setError(null);
      setGenerationStats(null);

      revokeObjectUrlIfNeeded(audioUrl);
      setAudioUrl(null);

      const record = await api.createVoiceCloningRecord({
        text: text.trim(),
        model_id: selectedModel,
        language: language === "Auto" ? undefined : language,
        max_tokens: 0,
        reference_audio: voiceCloneAudio,
        reference_text: voiceCloneTranscript,
      });

      setAudioUrl(api.voiceCloningRecordAudioUrl(record.id));
      setGenerationStats(mapRecordToStats(record));
      setLatestRecord(record);

      setTimeout(() => {
        audioRef.current?.play().catch(() => {});
      }, 100);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setGenerating(false);
    }
  };

  const handleStop = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  };

  const handleDownload = async () => {
    const record = latestRecord;
    const localAudioUrl = !record ? audioUrl : null;
    if ((!record && !localAudioUrl) || isDownloading) {
      return;
    }

    beginDownload();
    try {
      if (record) {
        const downloadUrl = api.voiceCloningRecordAudioUrl(record.id, {
          download: true,
        });
        const filename =
          record.audio_filename || `izwi-voice-clone-${Date.now()}.wav`;
        await api.downloadAudioFile(downloadUrl, filename);
        completeDownload();
        return;
      }

      if (!localAudioUrl) {
        return;
      }
      await api.downloadAudioFile(
        localAudioUrl,
        `izwi-voice-clone-${Date.now()}.wav`,
      );
      completeDownload();
    } catch (error) {
      failDownload(error);
    }
  };

  const handleReset = () => {
    setText(DEFAULT_AUDITION_TEXT);
    setError(null);
    setGenerationStats(null);
    revokeObjectUrlIfNeeded(audioUrl);
    setAudioUrl(null);
    textareaRef.current?.focus();
  };

  const handleVoiceCloneReady = (audio: string, transcript: string) => {
    setVoiceCloneAudio(audio);
    setVoiceCloneTranscript(transcript);
    setIsVoiceReady(true);
    setSavedVoiceId(null);
    setError(null);
  };

  const handleVoiceCloneClear = () => {
    setVoiceCloneAudio(null);
    setVoiceCloneTranscript(null);
    setIsVoiceReady(false);
    setReferenceState(null);
    setSavedVoiceId(null);
    setConsentConfirmed(false);
  };

  const reusableVoiceId =
    savedVoiceId || referenceState?.activeSavedVoiceId || null;

  const getStatusTone = (option: ModelOption): string => {
    if (option.isReady) {
      return "text-[var(--text-secondary)] bg-[var(--bg-surface-3)] border border-[var(--border-muted)]";
    }
    if (
      option.statusLabel.toLowerCase().includes("downloading") ||
      option.statusLabel.toLowerCase().includes("loading")
    ) {
      return "text-amber-400 bg-amber-500/10";
    }
    if (option.statusLabel.toLowerCase().includes("error")) {
      return "text-red-400 bg-red-500/10";
    }
    return "text-[var(--text-muted)] bg-[var(--bg-surface-2)] border border-[var(--border-muted)]";
  };

  const handleOpenModels = () => {
    setIsModelMenuOpen(false);
    onOpenModelManager?.();
  };

  const renderModelSelector = () => (
    <div
      className="relative inline-block w-[280px] max-w-[85vw]"
      ref={modelMenuRef}
    >
      <button
        onClick={() => setIsModelMenuOpen((prev) => !prev)}
        className={clsx(
          "h-9 w-full px-3 rounded-lg border inline-flex items-center justify-between gap-2 text-xs transition-colors",
          selectedOption?.isReady
            ? "border-[var(--border-strong)] bg-[var(--bg-surface-3)] text-[var(--text-primary)]"
            : "border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] hover:border-[var(--border-strong)]",
        )}
      >
        <span className="flex-1 min-w-0 truncate text-left">
          {selectedOption?.label || "Select model"}
        </span>
        <ChevronDown
          className={clsx(
            "w-3.5 h-3.5 shrink-0 transition-transform",
            isModelMenuOpen && "rotate-180",
          )}
        />
      </button>

      <AnimatePresence>
        {isModelMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            transition={{ duration: 0.15 }}
            className="absolute left-0 right-0 top-full mt-2 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-2)] p-1.5 shadow-2xl z-50"
          >
            <div className="max-h-64 overflow-y-auto pr-1 space-y-0.5">
              {modelOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => {
                    onSelectModel?.(option.value);
                    setIsModelMenuOpen(false);
                  }}
                  className={clsx(
                    "w-full text-left rounded-lg px-3 py-2 transition-colors",
                    selectedOption?.value === option.value
                      ? "bg-[var(--bg-surface-3)]"
                      : "hover:bg-[var(--bg-surface-3)]",
                  )}
                >
                  <div className="text-xs text-[var(--text-primary)] truncate">
                    {option.label}
                  </div>
                  <span
                    className={clsx(
                      "mt-1 inline-flex items-center rounded px-1.5 py-0.5 text-[10px]",
                      getStatusTone(option),
                    )}
                  >
                    {option.statusLabel}
                  </span>
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );

  return (
    <div className="grid gap-6 items-stretch xl:h-[calc(100dvh-11.75rem)]">
      <div className="card p-6 flex min-h-0 flex-col">
        <div className="flex-1 min-h-0 overflow-y-auto pr-1 scrollbar-thin">
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-6 gap-4">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-[var(--bg-surface-2)]">
                <Users className="w-5 h-5 text-[var(--text-primary)]" />
              </div>
              <div>
                <h2 className={VOICE_ROUTE_WORKSPACE_TITLE_CLASS}>
                  Voice Cloning
                </h2>
                <p className={VOICE_ROUTE_WORKSPACE_DESCRIPTION_CLASS}>
                  Prepare a reference, confirm quality, and audition a reusable
                  cloned voice before sending it to TTS.
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-end gap-2">
              <div className="relative w-full sm:w-auto">
                <button
                  onClick={() => setShowLanguageSelect(!showLanguageSelect)}
                  className="flex w-full sm:w-56 items-center justify-between gap-2 px-3 py-2 rounded-lg bg-[var(--bg-surface-2)] border border-[var(--border-muted)] hover:border-[var(--border-strong)] transition-colors text-sm"
                >
                  <Globe className="w-4 h-4 text-[var(--text-muted)]" />
                  <span className="text-[var(--text-primary)] font-medium flex-1 min-w-0 truncate text-left">
                    {LANGUAGES.find((l) => l.id === language)?.name || language}
                  </span>
                  <ChevronDown
                    className={clsx(
                      "w-4 h-4 text-[var(--text-muted)] transition-transform",
                      showLanguageSelect && "rotate-180",
                    )}
                  />
                </button>

                <AnimatePresence>
                  {showLanguageSelect && (
                    <motion.div
                      initial={{ opacity: 0, y: -4 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -4 }}
                      transition={{ duration: 0.15 }}
                      className="absolute left-0 right-0 top-full mt-2 max-h-64 overflow-y-auto p-1.5 rounded-xl bg-[var(--bg-surface-1)] border border-[var(--border-strong)] shadow-xl z-50"
                    >
                      {LANGUAGES.map((lang) => (
                        <button
                          key={lang.id}
                          onClick={() => {
                            setLanguage(lang.id);
                            setShowLanguageSelect(false);
                          }}
                          className={clsx(
                            "w-full px-3 py-2.5 rounded-lg text-left text-sm font-medium transition-colors",
                            language === lang.id
                              ? "bg-[var(--bg-surface-2)] text-[var(--text-primary)]"
                              : "hover:bg-[var(--bg-surface-2)] text-[var(--text-secondary)]",
                          )}
                        >
                          {lang.name}
                        </button>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          </div>

          <div className="mb-6 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 sm:p-5">
            <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
              <div className="min-w-0 flex-1">
                <div className={clsx(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-2.5")}>
                  Active Model
                </div>
                {modelOptions.length > 0 && <div>{renderModelSelector()}</div>}
                <div
                  className={clsx(
                    "mt-2.5 flex items-center gap-1.5 text-xs font-medium",
                    selectedModelReady
                      ? "text-green-500"
                      : "text-[var(--text-muted)]",
                  )}
                >
                  {selectedModelReady ? (
                    <>
                      <div className="w-1.5 h-1.5 rounded-full bg-green-500" />
                      Ready for cloning
                    </>
                  ) : (
                    <>
                      <div className="w-1.5 h-1.5 rounded-full bg-[var(--text-muted)]" />
                      Select a downloaded model to begin
                    </>
                  )}
                </div>
              </div>
              {onOpenModelManager && (
                <div className="shrink-0 mt-2 sm:mt-0">
                  <button
                    onClick={handleOpenModels}
                    className="btn btn-secondary text-xs h-9 px-4 rounded-md"
                  >
                    <Settings2 className="w-4 h-4" />
                    Manage
                  </button>
                </div>
              )}
            </div>
          </div>

          <div className="space-y-6">
            {/* Voice Reference Section */}
            <div className="p-6 rounded-2xl bg-[var(--bg-surface-0)] border border-[var(--border-muted)]">
              <div className="flex items-center gap-2 mb-5">
                <Users className="w-5 h-5 text-[var(--text-muted)]" />
                <span className="text-sm font-semibold text-[var(--text-primary)]">
                  Voice Reference
                </span>
                {isVoiceReady && (
                  <span className="text-[10px] font-medium px-2 py-0.5 rounded-md bg-green-500/10 text-green-500 border border-green-500/20 ml-2">
                    Ready
                  </span>
                )}
              </div>
              <VoiceClone
                onVoiceCloneReady={handleVoiceCloneReady}
                onClear={handleVoiceCloneClear}
                onReferenceStateChange={setReferenceState}
                onSavedVoiceCreated={setSavedVoiceId}
              />

              <div className="mt-5 grid gap-4 xl:grid-cols-2">
                <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-5">
                  <div className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-3">
                    Quality Checks
                  </div>
                  <div className="space-y-2.5 text-sm">
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-[var(--text-secondary)]">
                        Reference length
                      </span>
                      <span className="font-medium text-[var(--text-primary)]">
                        {referenceState?.sampleDurationSecs
                          ? `${referenceState.sampleDurationSecs.toFixed(1)}s`
                          : "No sample yet"}
                      </span>
                    </div>
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-[var(--text-secondary)]">
                        Transcript coverage
                      </span>
                      <span className="font-medium text-[var(--text-primary)]">
                        {referenceState?.transcriptChars ?? 0} chars
                      </span>
                    </div>
                    {!referenceState?.sampleReady ? (
                      <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2 text-xs text-[var(--text-muted)]">
                        Add a clean reference sample and transcript to see
                        cloning guidance.
                      </div>
                    ) : referenceState.warnings?.length ? (
                      <div className="rounded-lg border border-amber-500/25 bg-amber-500/10 px-3 py-2 text-xs text-amber-700">
                        {referenceState.warnings[0]}
                      </div>
                    ) : (
                      <div className="rounded-lg border border-green-500/25 bg-green-500/10 px-3 py-2 text-xs text-green-600">
                        Clean sample and transcript are ready for cloning.
                      </div>
                    )}
                  </div>
                </div>

                <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
                  <div className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                    Rights and Reuse
                  </div>
                  <label className="flex items-start gap-3 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-3 text-sm">
                    <input
                      type="checkbox"
                      checked={consentConfirmed}
                      onChange={(event) =>
                        setConsentConfirmed(event.target.checked)
                      }
                      className="mt-0.5 h-4 w-4 rounded border-[var(--border-muted)]"
                    />
                    <span className="text-[var(--text-secondary)]">
                      I have permission to clone this voice and use the
                      resulting audio.
                    </span>
                  </label>

                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    {reusableVoiceId && onUseInTts ? (
                      <button
                        onClick={() => onUseInTts(reusableVoiceId)}
                        className="btn btn-secondary h-9 px-4 rounded-lg text-xs"
                      >
                        <ArrowRight className="w-4 h-4" />
                        Use in TTS
                      </button>
                    ) : (
                      <div className="text-xs text-[var(--text-muted)]">
                        Save the reference as a voice profile to reuse it
                        directly in text-to-speech.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Text to speak */}
            <div className="p-6 rounded-2xl bg-[var(--bg-surface-0)] border border-[var(--border-muted)]">
              <label className="block text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-4">
                Audition Text
              </label>
              <div className="relative">
                <textarea
                  ref={textareaRef}
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Write a short preview line to audition the cloned voice..."
                  rows={5}
                  disabled={generating}
                  className="textarea text-sm py-4 px-5 leading-relaxed bg-[var(--bg-surface-1)] border-[var(--border-muted)] w-full rounded-xl"
                />
                <div className="absolute bottom-4 right-4 px-1">
                  <span className="text-[11px] font-medium text-[var(--text-muted)]">
                    {text.length} characters
                  </span>
                </div>
              </div>
              <p className="mt-4 text-xs leading-relaxed text-[var(--text-muted)]">
                Start with a short proof clip before moving the saved voice into
                the TTS route for longer scripts.
              </p>
            </div>

            {/* Error */}
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="p-3 rounded-lg bg-[var(--danger-bg)] border border-[var(--danger-border)] text-sm text-[var(--danger-text)] flex items-start gap-2 mt-2">
                    <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                    <p>{error}</p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Actions */}
            <div className="flex items-center gap-3 flex-wrap sm:flex-nowrap pt-2">
              <button
                onClick={handleGenerate}
                disabled={
                  generating ||
                  !selectedModelReady ||
                  !isVoiceReady ||
                  !consentConfirmed
                }
                className="btn btn-primary flex-1 h-11 text-sm font-semibold rounded-lg"
              >
                {generating ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Creating Preview...
                  </>
                ) : (
                  <>
                    <BadgeCheck className="w-4 h-4" />
                    Create Preview
                  </>
                )}
              </button>

              {audioUrl && (
                <>
                  <button
                    onClick={handleStop}
                    className="btn btn-secondary h-11 w-11 p-0 rounded-lg shrink-0"
                    title="Stop playback"
                  >
                    <Square className="w-4 h-4" />
                  </button>
                  <button
                    onClick={handleDownload}
                    disabled={isDownloading}
                    className={clsx(
                      "btn btn-secondary h-11 w-11 p-0 rounded-lg shrink-0",
                      isDownloading && "opacity-75",
                    )}
                    title="Download audio"
                  >
                    {isDownloading ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Download className="w-4 h-4" />
                    )}
                  </button>
                  <button
                    onClick={handleReset}
                    className="btn btn-ghost h-11 w-11 p-0 rounded-lg shrink-0 border border-transparent hover:border-[var(--border-muted)]"
                    title="Reset form"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>
                </>
              )}
            </div>

            <AnimatePresence>
              {downloadState !== "idle" && downloadMessage && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden mt-3"
                >
                  <div
                    className={clsx(
                      "px-3 py-2.5 rounded-lg border text-xs font-medium flex items-center gap-2",
                      downloadState === "downloading" &&
                        "bg-[var(--status-warning-bg)] border-[var(--status-warning-border)] text-[var(--status-warning-text)]",
                      downloadState === "success" &&
                        "bg-[var(--status-positive-bg)] border-[var(--status-positive-border)] text-[var(--status-positive-text)]",
                      downloadState === "error" &&
                        "bg-[var(--danger-bg)] border-[var(--danger-border)] text-[var(--danger-text)]",
                    )}
                  >
                    {downloadState === "downloading" ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : downloadState === "success" ? (
                      <CheckCircle2 className="w-3.5 h-3.5" />
                    ) : (
                      <AlertCircle className="w-3.5 h-3.5" />
                    )}
                    {downloadMessage}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {!selectedModelReady && (
              <p className="text-xs font-medium text-[var(--text-muted)] text-center pb-2">
                Load a Base model to clone voices
              </p>
            )}

            {selectedModelReady && !isVoiceReady && (
              <p className="text-xs font-medium text-[var(--text-muted)] text-center pb-2">
                Upload, record, or select a saved voice sample to get started
              </p>
            )}

            {selectedModelReady && isVoiceReady && !consentConfirmed && (
              <p className="text-xs font-medium text-[var(--text-muted)] text-center pb-2">
                Confirm rights to the reference voice before generating a
                preview.
              </p>
            )}
          </div>

          {/* Audio player */}
          <AnimatePresence>
            {audioUrl && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="mt-6 space-y-4"
              >
                <div className="p-4 rounded-xl bg-[var(--bg-surface-2)] border border-[var(--border-strong)] shadow-sm">
                  <audio
                    ref={audioRef}
                    src={audioUrl}
                    className="w-full h-11"
                    controls
                  />
                </div>
                {generationStats && (
                  <GenerationStats stats={generationStats} type="tts" />
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      <SpeechHistoryPanel
        route="voice-cloning"
        title="Voice Cloning History"
        emptyMessage="No saved voice-cloning generations yet."
        latestRecord={latestRecord}
        historyActionContainer={historyActionContainer}
      />
    </div>
  );
}
