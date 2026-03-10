import { useState, useRef, useMemo, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Wand2,
  Square,
  Download,
  RotateCcw,
  BookmarkPlus,
  ChevronDown,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Globe,
  Settings2,
} from "lucide-react";
import { api, type SpeechHistoryRecord, type TTSGenerationStats } from "../api";
import { LANGUAGES, VOICE_DESIGN_PRESETS } from "../types";
import clsx from "clsx";
import { GenerationStats } from "./GenerationStats";
import { SpeechHistoryPanel } from "./SpeechHistoryPanel";
import { useDownloadIndicator } from "../utils/useDownloadIndicator";
import { blobToBase64Payload } from "../utils/audioBase64";

interface ModelOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

interface VoiceDesignPlaygroundProps {
  selectedModel: string | null;
  selectedModelReady?: boolean;
  modelOptions?: ModelOption[];
  onSelectModel?: (variant: string) => void;
  onOpenModelManager?: () => void;
  onModelRequired: () => void;
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

export function VoiceDesignPlayground({
  selectedModel,
  selectedModelReady = false,
  modelOptions = [],
  onSelectModel,
  onOpenModelManager,
  onModelRequired,
  historyActionContainer = null,
}: VoiceDesignPlaygroundProps) {
  const [text, setText] = useState("");
  const [voiceDescription, setVoiceDescription] = useState("");
  const [language, setLanguage] = useState("Auto");
  const [showLanguageSelect, setShowLanguageSelect] = useState(false);
  const [showPresets, setShowPresets] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [generationStats, setGenerationStats] =
    useState<TTSGenerationStats | null>(null);
  const [latestRecord, setLatestRecord] = useState<SpeechHistoryRecord | null>(
    null,
  );
  const [saveVoiceName, setSaveVoiceName] = useState("");
  const [saveReferenceText, setSaveReferenceText] = useState("");
  const [savingVoice, setSavingVoice] = useState(false);
  const [saveVoiceStatus, setSaveVoiceStatus] = useState<{
    tone: "success" | "error";
    message: string;
  } | null>(null);
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

  useEffect(() => {
    if (!latestRecord) {
      return;
    }
    setSaveVoiceName("");
    setSaveReferenceText(latestRecord.input_text);
    setSaveVoiceStatus(null);
  }, [latestRecord?.id, latestRecord?.input_text]);

  const handleGenerate = async () => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      return;
    }

    if (!text.trim()) {
      setError("Please enter some text to synthesize");
      return;
    }

    if (!voiceDescription.trim()) {
      setError("Please describe the voice you want to create");
      return;
    }

    try {
      setGenerating(true);
      setError(null);
      setGenerationStats(null);
      setSaveVoiceStatus(null);

      revokeObjectUrlIfNeeded(audioUrl);
      setAudioUrl(null);

      const record = await api.createVoiceDesignRecord({
        text: text.trim(),
        model_id: selectedModel,
        language: language === "Auto" ? undefined : language,
        max_tokens: 0,
        voice_description: voiceDescription.trim(),
      });

      setAudioUrl(api.voiceDesignRecordAudioUrl(record.id));
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
        const downloadUrl = api.voiceDesignRecordAudioUrl(record.id, {
          download: true,
        });
        const filename =
          record.audio_filename || `izwi-voice-design-${Date.now()}.wav`;
        await api.downloadAudioFile(downloadUrl, filename);
        completeDownload();
        return;
      }

      if (!localAudioUrl) {
        return;
      }
      await api.downloadAudioFile(
        localAudioUrl,
        `izwi-voice-design-${Date.now()}.wav`,
      );
      completeDownload();
    } catch (error) {
      failDownload(error);
    }
  };

  const handleReset = () => {
    setText("");
    setVoiceDescription("");
    setError(null);
    setGenerationStats(null);
    revokeObjectUrlIfNeeded(audioUrl);
    setAudioUrl(null);
    textareaRef.current?.focus();
  };

  const handlePresetSelect = (description: string) => {
    setVoiceDescription(description);
    setShowPresets(false);
  };

  const handleSaveVoice = async () => {
    if (!latestRecord || savingVoice) {
      return;
    }

    const trimmedName = saveVoiceName.trim();
    if (!trimmedName) {
      setSaveVoiceStatus({
        tone: "error",
        message: "Enter a name before saving this voice.",
      });
      return;
    }

    const trimmedReferenceText = saveReferenceText.trim();
    if (!trimmedReferenceText) {
      setSaveVoiceStatus({
        tone: "error",
        message: "Reference text is required for voice cloning.",
      });
      return;
    }

    setSavingVoice(true);
    setSaveVoiceStatus(null);

    try {
      const response = await fetch(
        api.voiceDesignRecordAudioUrl(latestRecord.id),
      );
      if (!response.ok) {
        throw new Error(`Failed to load generated audio (${response.status})`);
      }

      const audioBlob = await response.blob();
      const audioBase64 = await blobToBase64Payload(audioBlob);

      await api.createSavedVoice({
        name: trimmedName,
        reference_text: trimmedReferenceText,
        audio_base64: audioBase64,
        audio_mime_type:
          latestRecord.audio_mime_type || audioBlob.type || "audio/wav",
        audio_filename:
          latestRecord.audio_filename || `voice-design-saved-${Date.now()}.wav`,
        source_route_kind: "voice_design",
        source_record_id: latestRecord.id,
      });

      setSaveVoiceName("");
      setSaveVoiceStatus({
        tone: "success",
        message: `Saved voice profile "${trimmedName}".`,
      });
    } catch (err) {
      setSaveVoiceStatus({
        tone: "error",
        message:
          err instanceof Error ? err.message : "Failed to save voice profile.",
      });
    } finally {
      setSavingVoice(false);
    }
  };

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
    <div className="grid gap-4 items-stretch xl:h-[calc(100dvh-11.75rem)]">
      <div className="card p-4 flex min-h-0 flex-col">
        <div className="flex-1 min-h-0 overflow-y-auto pr-1 scrollbar-thin">
          {/* Header */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded bg-[var(--bg-surface-2)] border border-[var(--border-muted)]">
                <Wand2 className="w-5 h-5 text-[var(--text-muted)]" />
              </div>
              <div>
                <h2 className="text-sm font-medium text-[var(--text-primary)]">
                  Voice Design
                </h2>
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-end gap-2">
              <div className="relative">
                <button
                  onClick={() => setShowLanguageSelect(!showLanguageSelect)}
                  className="flex w-52 sm:w-56 items-center justify-between gap-2 px-3 py-1.5 rounded bg-[var(--bg-surface-2)] border border-[var(--border-muted)] hover:bg-[var(--bg-surface-3)] text-sm"
                >
                  <Globe className="w-3.5 h-3.5 text-[var(--text-subtle)]" />
                  <span className="text-[var(--text-primary)] flex-1 min-w-0 truncate text-left">
                    {LANGUAGES.find((l) => l.id === language)?.name || language}
                  </span>
                  <ChevronDown
                    className={clsx(
                      "w-3.5 h-3.5 text-[var(--text-subtle)] transition-transform",
                      showLanguageSelect && "rotate-180",
                    )}
                  />
                </button>

                <AnimatePresence>
                  {showLanguageSelect && (
                    <motion.div
                      initial={{ opacity: 0, y: -5 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -5 }}
                      className="absolute left-0 right-0 top-full mt-1 max-h-64 overflow-y-auto p-1 rounded bg-[var(--bg-surface-2)] border border-[var(--border-muted)] shadow-xl z-50"
                    >
                      {LANGUAGES.map((lang) => (
                        <button
                          key={lang.id}
                          onClick={() => {
                            setLanguage(lang.id);
                            setShowLanguageSelect(false);
                          }}
                          className={clsx(
                            "w-full px-2 py-1.5 rounded text-left text-sm transition-colors",
                            language === lang.id
                              ? "bg-[var(--bg-surface-3)] text-[var(--text-primary)]"
                              : "hover:bg-[var(--bg-surface-3)] text-[var(--text-secondary)]",
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

          <div className="mb-4 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0 flex-1">
                <div className="text-[11px] text-[var(--text-subtle)] uppercase tracking-wide">
                  Active Model
                </div>
                {modelOptions.length > 0 && (
                  <div className="mt-2">{renderModelSelector()}</div>
                )}
                <div
                  className={clsx(
                    "mt-2 text-xs",
                    selectedModelReady
                      ? "text-[var(--text-secondary)]"
                      : "text-amber-400",
                  )}
                >
                  {selectedModelReady
                    ? "Loaded and ready"
                    : "Open Models and load a VoiceDesign model"}
                </div>
              </div>
              {onOpenModelManager && (
                <div className="shrink-0">
                  <button
                    onClick={handleOpenModels}
                    className="btn btn-secondary text-xs"
                  >
                    <Settings2 className="w-4 h-4" />
                    Models
                  </button>
                </div>
              )}
            </div>
          </div>

          <div className="space-y-6">
            {/* Voice Description */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wide">
                  Voice Description
                  <span className="text-red-500 ml-1">*</span>
                </label>
                <button
                  onClick={() => setShowPresets(!showPresets)}
                  className="text-xs font-medium text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
                >
                  {showPresets ? "Hide presets" : "View presets"}
                </button>
              </div>

              <AnimatePresence>
                {showPresets && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mb-3 overflow-hidden"
                  >
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 p-3 rounded-xl bg-[var(--bg-surface-1)] border border-[var(--border-muted)]">
                      {VOICE_DESIGN_PRESETS.map((preset) => (
                        <button
                          key={preset.name}
                          onClick={() => handlePresetSelect(preset.description)}
                          className="p-3 rounded-lg bg-[var(--bg-surface-0)] hover:bg-[var(--bg-surface-2)] border border-[var(--border-muted)] hover:border-[var(--border-strong)] text-left transition-colors group"
                        >
                          <div className="text-sm font-semibold text-[var(--text-primary)] mb-1">
                            {preset.name}
                          </div>
                          <div className="text-[11px] text-[var(--text-secondary)] line-clamp-2 leading-relaxed group-hover:text-[var(--text-primary)] transition-colors">
                            {preset.description}
                          </div>
                        </button>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              <textarea
                value={voiceDescription}
                onChange={(e) => setVoiceDescription(e.target.value)}
                placeholder="Describe the voice you want to create... (e.g., 'A warm, friendly female voice with a slight British accent, speaking in a calm and reassuring tone')"
                rows={4}
                className="textarea text-base py-4 leading-relaxed bg-[var(--bg-surface-0)] border-[var(--border-muted)] w-full"
              />
              <p className="text-[11px] font-medium text-[var(--text-muted)] mt-2">
                Describe voice characteristics like gender, age, tone, emotion,
                accent, and speaking style.
              </p>
            </div>

            {/* Text to speak */}
            <div>
              <label className="block text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wide mb-2">
                Text to Speak
                <span className="text-red-500 ml-1">*</span>
              </label>
              <div className="relative">
                <textarea
                  ref={textareaRef}
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Enter the text you want the designed voice to synthesize..."
                  rows={4}
                  disabled={generating}
                  className="textarea text-base py-4 leading-relaxed bg-[var(--bg-surface-1)] border-[var(--border-muted)] w-full"
                />
                <div className="absolute bottom-3 right-3 px-1">
                  <span className="text-[11px] font-medium text-[var(--text-muted)]">
                    {text.length} characters
                  </span>
                </div>
              </div>
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
                disabled={generating || !selectedModelReady}
                className="btn btn-primary flex-1 h-11 text-sm font-semibold rounded-lg"
              >
                {generating ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Designing Voice...
                  </>
                ) : (
                  <>
                    <Wand2 className="w-4 h-4" />
                    Generate Audio
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
                Load a VoiceDesign model to create unique voices
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
                {latestRecord && (
                  <div className="p-4 rounded-xl bg-[var(--bg-surface-1)] border border-[var(--border-muted)] space-y-4">
                    <div className="flex items-center justify-between gap-2">
                      <div className="text-sm font-semibold text-[var(--text-primary)]">
                        Save for Voice Cloning
                      </div>
                      <span className="text-[10px] font-medium px-2 py-0.5 rounded-md bg-[var(--bg-surface-2)] text-[var(--text-muted)]">
                        Reuse on /voice-clone
                      </span>
                    </div>

                    <div className="space-y-4">
                      <div>
                        <label className="block text-[11px] font-medium text-[var(--text-secondary)] mb-1.5">
                          Voice Name
                        </label>
                        <input
                          value={saveVoiceName}
                          onChange={(event) =>
                            setSaveVoiceName(event.target.value)
                          }
                          placeholder="e.g. Support Voice"
                          className="input h-10 text-sm bg-[var(--bg-surface-0)] border-[var(--border-muted)] w-full"
                          disabled={savingVoice}
                        />
                      </div>

                      <div>
                        <label className="block text-[11px] font-medium text-[var(--text-secondary)] mb-1.5">
                          Reference Transcript
                        </label>
                        <textarea
                          value={saveReferenceText}
                          onChange={(event) =>
                            setSaveReferenceText(event.target.value)
                          }
                          rows={2}
                          className="textarea text-sm bg-[var(--bg-surface-0)] border-[var(--border-muted)] w-full"
                          disabled={savingVoice}
                          placeholder="Reference transcript for cloning"
                        />
                        <p className="text-[10px] text-[var(--text-subtle)] mt-1.5">
                          Keep this transcript exactly aligned with the
                          generated audio sample.
                        </p>
                      </div>

                      <div className="flex justify-end pt-2">
                        <button
                          onClick={handleSaveVoice}
                          disabled={savingVoice || !saveVoiceName.trim()}
                          className="btn btn-secondary h-10 px-5 gap-2 border-[var(--border-muted)]"
                        >
                          {savingVoice ? (
                            <>
                              <Loader2 className="w-4 h-4 animate-spin" />
                              Saving...
                            </>
                          ) : (
                            <>
                              <BookmarkPlus className="w-4 h-4" />
                              Save Profile
                            </>
                          )}
                        </button>
                      </div>
                    </div>

                    <AnimatePresence>
                      {saveVoiceStatus && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          exit={{ opacity: 0, height: 0 }}
                          className="overflow-hidden"
                        >
                          <div
                            className={clsx(
                              "mt-2 p-3 rounded-lg border text-xs font-medium flex items-center gap-2",
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
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      <SpeechHistoryPanel
        route="voice-design"
        title="Voice Design History"
        emptyMessage="No saved voice design generations yet."
        latestRecord={latestRecord}
        historyActionContainer={historyActionContainer}
      />
    </div>
  );
}
