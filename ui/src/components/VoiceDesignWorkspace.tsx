import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertCircle,
  ArrowRight,
  BookmarkPlus,
  CheckCircle2,
  ChevronDown,
  Download,
  Globe,
  Loader2,
  Settings2,
  Sparkles,
  Wand2,
} from "lucide-react";
import clsx from "clsx";
import { api, type SpeechHistoryRecord, type TTSGenerationStats } from "../api";
import { LANGUAGES, VOICE_DESIGN_PRESETS } from "../types";
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

interface VoiceDesignWorkspaceProps {
  selectedModel: string | null;
  selectedModelReady?: boolean;
  modelOptions?: ModelOption[];
  onSelectModel?: (variant: string) => void;
  onOpenModelManager?: () => void;
  onModelRequired: () => void;
  onUseInTts?: (voiceId: string) => void;
  historyActionContainer?: HTMLElement | null;
}

interface VoiceDesignCandidate {
  id: string;
  label: string;
  nuance: string;
  prompt: string;
  record: SpeechHistoryRecord;
  stats: TTSGenerationStats;
}

const DEFAULT_SAMPLE_TEXT =
  "Hello, this is Izwi. This short preview helps you compare the voice before using it in a larger text-to-speech workflow.";

const CANDIDATE_VARIATIONS = [
  {
    id: "balanced",
    label: "Balanced",
    nuance: "Keeps the original direction with minimal adjustment.",
    suffix: "",
  },
  {
    id: "warm",
    label: "Warm",
    nuance: "Pushes the delivery toward warmth and conversational energy.",
    suffix: " Make the delivery slightly warmer, more human, and more conversational.",
  },
  {
    id: "polished",
    label: "Polished",
    nuance: "Pushes the delivery toward precision, confidence, and authority.",
    suffix:
      " Make the delivery slightly more polished, authoritative, and presentation-ready.",
  },
] as const;

function mapRecordToStats(record: SpeechHistoryRecord): TTSGenerationStats {
  return {
    generation_time_ms: record.generation_time_ms,
    audio_duration_secs: record.audio_duration_secs ?? 0,
    rtf: record.rtf ?? 0,
    tokens_generated: record.tokens_generated ?? 0,
  };
}

export function VoiceDesignWorkspace({
  selectedModel,
  selectedModelReady = false,
  modelOptions = [],
  onSelectModel,
  onOpenModelManager,
  onModelRequired,
  onUseInTts,
  historyActionContainer = null,
}: VoiceDesignWorkspaceProps) {
  const [text, setText] = useState(DEFAULT_SAMPLE_TEXT);
  const [voiceDescription, setVoiceDescription] = useState("");
  const [language, setLanguage] = useState("Auto");
  const [showLanguageSelect, setShowLanguageSelect] = useState(false);
  const [showPresets, setShowPresets] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [generationStep, setGenerationStep] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<VoiceDesignCandidate[]>([]);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(
    null,
  );
  const [latestRecord, setLatestRecord] = useState<SpeechHistoryRecord | null>(
    null,
  );
  const [saveVoiceName, setSaveVoiceName] = useState("");
  const [savingVoice, setSavingVoice] = useState(false);
  const [savedVoiceId, setSavedVoiceId] = useState<string | null>(null);
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

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);

  const selectedOption = useMemo(() => {
    if (!selectedModel) {
      return null;
    }
    return modelOptions.find((option) => option.value === selectedModel) || null;
  }, [modelOptions, selectedModel]);

  const selectedCandidate = useMemo(
    () => candidates.find((candidate) => candidate.id === selectedCandidateId) ?? null,
    [candidates, selectedCandidateId],
  );

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
    if (!selectedCandidate) {
      setSaveVoiceName("");
      setSavedVoiceId(null);
      return;
    }
    setSaveVoiceName((current) =>
      current.trim() ? current : `${selectedCandidate.label} voice`,
    );
    setSavedVoiceId(null);
    setSaveVoiceStatus(null);
  }, [selectedCandidate?.id]);

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

  const handlePresetSelect = (description: string) => {
    setVoiceDescription(description);
    setShowPresets(false);
  };

  const handleGenerate = async () => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      return;
    }

    if (!voiceDescription.trim()) {
      setError("Describe the voice you want to design before generating options.");
      return;
    }

    if (!text.trim()) {
      setError("Enter preview text before generating voice candidates.");
      return;
    }

    try {
      setGenerating(true);
      setGenerationStep("");
      setError(null);
      setCandidates([]);
      setSelectedCandidateId(null);
      setSaveVoiceStatus(null);
      setSavedVoiceId(null);

      const nextCandidates: VoiceDesignCandidate[] = [];
      for (const variation of CANDIDATE_VARIATIONS) {
        setGenerationStep(`Generating ${variation.label.toLowerCase()} option...`);
        const record = await api.createVoiceDesignRecord({
          text: text.trim(),
          model_id: selectedModel,
          language: language === "Auto" ? undefined : language,
          max_tokens: 0,
          voice_description: `${voiceDescription.trim()}${variation.suffix}`,
        });

        nextCandidates.push({
          id: variation.id,
          label: variation.label,
          nuance: variation.nuance,
          prompt: `${voiceDescription.trim()}${variation.suffix}`,
          record,
          stats: mapRecordToStats(record),
        });
        setCandidates([...nextCandidates]);
        setLatestRecord(record);
      }

      setSelectedCandidateId(nextCandidates[0]?.id ?? null);
      setGenerationStep("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Voice design failed");
      setGenerationStep("");
    } finally {
      setGenerating(false);
    }
  };

  const handleReset = () => {
    setText(DEFAULT_SAMPLE_TEXT);
    setVoiceDescription("");
    setError(null);
    setCandidates([]);
    setSelectedCandidateId(null);
    setGenerationStep("");
    setSaveVoiceName("");
    setSaveVoiceStatus(null);
    setSavedVoiceId(null);
    textareaRef.current?.focus();
  };

  const handleDownloadCandidate = async (candidate: VoiceDesignCandidate) => {
    if (isDownloading) {
      return;
    }
    beginDownload();
    try {
      const filename =
        candidate.record.audio_filename ||
        `izwi-voice-design-${candidate.label.toLowerCase()}-${Date.now()}.wav`;
      await api.downloadAudioFile(
        api.voiceDesignRecordAudioUrl(candidate.record.id, { download: true }),
        filename,
      );
      completeDownload();
    } catch (err) {
      failDownload(err);
    }
  };

  const handleSaveVoice = async () => {
    if (!selectedCandidate || savingVoice) {
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

    setSavingVoice(true);
    setSaveVoiceStatus(null);

    try {
      const response = await fetch(
        api.voiceDesignRecordAudioUrl(selectedCandidate.record.id),
      );
      if (!response.ok) {
        throw new Error(`Failed to load generated audio (${response.status})`);
      }

      const audioBlob = await response.blob();
      const audioBase64 = await blobToBase64Payload(audioBlob);
      const createdVoice = await api.createSavedVoice({
        name: trimmedName,
        reference_text: selectedCandidate.record.input_text,
        audio_base64: audioBase64,
        audio_mime_type:
          selectedCandidate.record.audio_mime_type ||
          audioBlob.type ||
          "audio/wav",
        audio_filename:
          selectedCandidate.record.audio_filename ||
          `voice-design-saved-${Date.now()}.wav`,
        source_route_kind: "voice_design",
        source_record_id: selectedCandidate.record.id,
      });

      setSavedVoiceId(createdVoice.id);
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

  return (
    <div className="grid gap-4 items-stretch xl:h-[calc(100dvh-11.75rem)]">
      <div className="card p-4 flex min-h-0 flex-col">
        <div className="flex-1 min-h-0 overflow-y-auto pr-1 scrollbar-thin">
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

          <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_420px]">
            <div className="space-y-6">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wide">
                    Voice Direction
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
                  onChange={(event) => setVoiceDescription(event.target.value)}
                  placeholder="Describe the voice you want to create. Include tone, age, use case, energy, accent, or delivery style."
                  rows={4}
                  className="textarea text-base py-4 leading-relaxed bg-[var(--bg-surface-0)] border-[var(--border-muted)] w-full"
                />
                <p className="text-[11px] font-medium text-[var(--text-muted)] mt-2">
                  We will generate three nearby directions so you can compare before saving a reusable voice.
                </p>
              </div>

              <div>
                <label className="block text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wide mb-2">
                  Preview Script
                  <span className="text-red-500 ml-1">*</span>
                </label>
                <div className="relative">
                  <textarea
                    ref={textareaRef}
                    value={text}
                    onChange={(event) => setText(event.target.value)}
                    placeholder="Enter the preview text you want each designed voice to speak..."
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

              <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-2">
                  Candidate Workflow
                </div>
                <div className="grid gap-2 sm:grid-cols-3 text-xs">
                  <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2">
                    1. Describe the voice
                  </div>
                  <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2">
                    2. Compare three options
                  </div>
                  <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2">
                    3. Save and open in TTS
                  </div>
                </div>
              </div>

              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="p-3 rounded-lg bg-[var(--danger-bg)] border border-[var(--danger-border)] text-sm text-[var(--danger-text)] flex items-start gap-2">
                      <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                      <p>{error}</p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="flex items-center gap-3 flex-wrap sm:flex-nowrap">
                <button
                  onClick={handleGenerate}
                  disabled={generating || !selectedModelReady}
                  className="btn btn-primary flex-1 h-11 text-sm font-semibold rounded-lg"
                >
                  {generating ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Generating options...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      Generate 3 options
                    </>
                  )}
                </button>

                <button
                  onClick={handleReset}
                  className="btn btn-ghost h-11 px-4 rounded-lg border border-transparent hover:border-[var(--border-muted)]"
                >
                  Reset
                </button>
              </div>

              {generationStep ? (
                <div className="text-xs font-medium text-[var(--text-muted)]">
                  {generationStep}
                </div>
              ) : null}

              {downloadState !== "idle" && downloadMessage ? (
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
              ) : null}
            </div>

            <div className="space-y-4">
              <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-3">
                  Candidate Compare
                </div>

                {candidates.length === 0 ? (
                  <div className="rounded-lg border border-dashed border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-8 text-center text-sm text-[var(--text-muted)]">
                    Generate options to compare designed voices side by side.
                  </div>
                ) : (
                  <div className="space-y-3">
                    {candidates.map((candidate) => {
                      const isSelected = candidate.id === selectedCandidateId;
                      return (
                        <div
                          key={candidate.id}
                          className={clsx(
                            "rounded-xl border p-4 transition-colors",
                            isSelected
                              ? "border-[var(--border-strong)] bg-[var(--bg-surface-0)]"
                              : "border-[var(--border-muted)] bg-[var(--bg-surface-0)]/60",
                          )}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <div className="text-sm font-semibold text-[var(--text-primary)]">
                                {candidate.label}
                              </div>
                              <div className="mt-1 text-xs text-[var(--text-secondary)]">
                                {candidate.nuance}
                              </div>
                            </div>
                            {isSelected ? (
                              <span className="rounded-full border border-[var(--border-strong)] bg-[var(--bg-surface-2)] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-[var(--text-primary)]">
                                Selected
                              </span>
                            ) : null}
                          </div>

                          <audio
                            src={api.voiceDesignRecordAudioUrl(candidate.record.id)}
                            controls
                            preload="none"
                            className="mt-3 h-10 w-full"
                          />

                          <div className="mt-3 grid gap-2 sm:grid-cols-2">
                            <button
                              onClick={() => setSelectedCandidateId(candidate.id)}
                              className="btn btn-secondary h-9 text-xs"
                            >
                              Select
                            </button>
                            <button
                              onClick={() => void handleDownloadCandidate(candidate)}
                              className="btn btn-ghost h-9 text-xs border border-[var(--border-muted)]"
                            >
                              <Download className="w-4 h-4" />
                              Download
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {selectedCandidate ? (
                <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 space-y-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold text-[var(--text-primary)]">
                        Save Selected Voice
                      </div>
                      <div className="text-xs text-[var(--text-secondary)] mt-1">
                        Turn the selected design into a reusable voice for TTS.
                      </div>
                    </div>
                    <span className="text-[10px] font-medium px-2 py-0.5 rounded-md bg-[var(--bg-surface-2)] text-[var(--text-muted)]">
                      {selectedCandidate.label}
                    </span>
                  </div>

                  <GenerationStats stats={selectedCandidate.stats} type="tts" />

                  <div>
                    <label className="block text-[11px] font-medium text-[var(--text-secondary)] mb-1.5">
                      Voice Name
                    </label>
                    <input
                      value={saveVoiceName}
                      onChange={(event) => setSaveVoiceName(event.target.value)}
                      placeholder="e.g. Support Voice"
                      className="input h-10 text-sm bg-[var(--bg-surface-0)] border-[var(--border-muted)] w-full"
                      disabled={savingVoice}
                    />
                  </div>

                  <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2 text-xs text-[var(--text-muted)]">
                    This voice will use the selected preview audio and its exact script as the reusable reference.
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
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
                          Save to My Voices
                        </>
                      )}
                    </button>

                    {savedVoiceId && onUseInTts ? (
                      <button
                        onClick={() => onUseInTts(savedVoiceId)}
                        className="btn btn-primary h-10 px-5 gap-2"
                      >
                        <ArrowRight className="w-4 h-4" />
                        Use in TTS
                      </button>
                    ) : null}
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
              ) : null}
            </div>
          </div>
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
