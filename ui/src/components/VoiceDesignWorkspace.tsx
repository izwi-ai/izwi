import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertCircle,
  ArrowRight,
  BookmarkPlus,
  CheckCircle2,
  Download,
  Globe,
  Loader2,
  Settings2,
  Sparkles,
  Wand2,
} from "lucide-react";
import clsx from "clsx";
import { api, type SpeechHistoryRecord, type TTSGenerationStats } from "../api";
import {
  VOICE_ROUTE_META_COPY_CLASS,
  VOICE_ROUTE_PANEL_TITLE_CLASS,
  VOICE_ROUTE_SECTION_LABEL_CLASS,
  VOICE_ROUTE_TITLE_ACCENT_CLASS,
} from "@/components/voiceRouteTypography";
import { LANGUAGES, VOICE_DESIGN_PRESETS } from "../types";
import { GenerationStats } from "./GenerationStats";
import { SpeechHistoryPanel } from "./SpeechHistoryPanel";
import { RouteModelSelect } from "@/components/RouteModelSelect";
import { StatePanel } from "@/components/ui/state-panel";
import { StatusBadge } from "@/components/ui/status-badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  WorkspaceHeader,
  WorkspacePanel,
} from "@/components/ui/workspace";
import { useWorkspaceShortcuts } from "@/hooks/useWorkspaceShortcuts";
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
  const {
    downloadState,
    downloadMessage,
    isDownloading,
    beginDownload,
    completeDownload,
    failDownload,
  } = useDownloadIndicator();

  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const selectedCandidate = useMemo(
    () => candidates.find((candidate) => candidate.id === selectedCandidateId) ?? null,
    [candidates, selectedCandidateId],
  );

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

  const handleOpenModels = () => {
    onOpenModelManager?.();
  };

  const renderModelSelector = () => (
    <RouteModelSelect
      value={selectedModel}
      options={modelOptions}
      onSelect={onSelectModel}
      className="w-full max-w-[280px]"
    />
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

  const workspaceShortcuts = useMemo(
    () => [
      {
        key: "Enter",
        metaKey: true,
        allowInInputs: true,
        enabled: !generating && selectedModelReady,
        action: () => {
          void handleGenerate();
        },
      },
      {
        key: "Escape",
        shiftKey: true,
        enabled: true,
        action: handleReset,
      },
    ],
    [generating, handleGenerate, handleReset, selectedModelReady],
  );

  useWorkspaceShortcuts(workspaceShortcuts);

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
    <div className="grid items-start gap-4 pb-4 sm:pb-5">
      <div className="flex flex-col">
        <WorkspaceHeader
          icon={Wand2}
          title="Voice Design"
          description="Describe a voice, compare nearby candidates, and save the best option for TTS reuse."
          className="border-none pb-0"
          actions={
            <Select value={language} onValueChange={setLanguage}>
              <SelectTrigger className="w-52 sm:w-56">
                <div className="flex min-w-0 items-center gap-2">
                  <Globe className="h-3.5 w-3.5 text-[var(--text-muted)]" />
                  <SelectValue placeholder="Language" />
                </div>
              </SelectTrigger>
              <SelectContent>
                {LANGUAGES.map((lang) => (
                  <SelectItem key={lang.id} value={lang.id}>
                    {lang.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          }
        />

        <WorkspacePanel className="mb-4 mt-5 p-4">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0 flex-1">
              <div className={VOICE_ROUTE_SECTION_LABEL_CLASS}>
                Active Model
              </div>
              {modelOptions.length > 0 && (
                <div className="mt-2">{renderModelSelector()}</div>
              )}
              <StatusBadge
                tone={selectedModelReady ? "success" : "warning"}
                className="mt-2"
              >
                {selectedModelReady
                  ? "Loaded and ready"
                  : "Open Models and load a VoiceDesign model"}
              </StatusBadge>
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
        </WorkspacePanel>

        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_420px]">
            <div className="space-y-6">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className={VOICE_ROUTE_SECTION_LABEL_CLASS}>
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
                            <div className={clsx(VOICE_ROUTE_TITLE_ACCENT_CLASS, "mb-1 text-sm")}>
                              {preset.name}
                            </div>
                            <div
                              className={clsx(
                                VOICE_ROUTE_META_COPY_CLASS,
                                "line-clamp-2 leading-relaxed group-hover:text-[var(--text-primary)] transition-colors",
                              )}
                            >
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
                <p className={clsx(VOICE_ROUTE_META_COPY_CLASS, "mt-2")}>
                  We will generate three nearby directions so you can compare before saving a reusable voice.
                </p>
              </div>

              <div>
                <label className={clsx(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-2 block")}>
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
                <div className={clsx(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-2")}>
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
                    <StatePanel
                      title="Voice design error"
                      description={error}
                      icon={AlertCircle}
                      tone="danger"
                    />
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

              <div className="text-xs text-[var(--text-muted)]">
                Shortcut: <span className="app-kbd">Ctrl/Cmd + Enter</span> generate options, <span className="app-kbd">Shift + Esc</span> reset.
              </div>
            </div>

            <div className="space-y-4">
              <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
                <div className={clsx(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-3")}>
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
                              <div className={clsx(VOICE_ROUTE_TITLE_ACCENT_CLASS, "text-sm")}>
                                {candidate.label}
                              </div>
                              <div className={clsx(VOICE_ROUTE_META_COPY_CLASS, "mt-1")}>
                                {candidate.nuance}
                              </div>
                            </div>
                            {isSelected ? (
                              <StatusBadge tone="info">Selected</StatusBadge>
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
                      <div className={VOICE_ROUTE_PANEL_TITLE_CLASS}>
                        Save Selected Voice
                      </div>
                      <div className={clsx(VOICE_ROUTE_META_COPY_CLASS, "mt-1")}>
                        Turn the selected design into a reusable voice for TTS.
                      </div>
                    </div>
                    <span className="text-[10px] font-medium px-2 py-0.5 rounded-md bg-[var(--bg-surface-2)] text-[var(--text-muted)]">
                      {selectedCandidate.label}
                    </span>
                  </div>

                  <GenerationStats stats={selectedCandidate.stats} type="tts" />

                  <div>
                    <label className="mb-1.5 block text-[11px] font-medium text-[var(--text-secondary)]">
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
