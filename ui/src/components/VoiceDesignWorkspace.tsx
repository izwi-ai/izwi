import { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  BookmarkPlus,
  CheckCircle2,
  Globe,
  Loader2,
  Settings2,
  Sparkles,
} from "lucide-react";
import clsx from "clsx";
import { api, type SpeechHistoryRecord } from "../api";
import {
  VOICE_ROUTE_META_COPY_CLASS,
  VOICE_ROUTE_PANEL_TITLE_CLASS,
  VOICE_ROUTE_SECTION_LABEL_CLASS,
  VOICE_ROUTE_TITLE_ACCENT_CLASS,
} from "@/components/voiceRouteTypography";
import { LANGUAGES, VOICE_DESIGN_PRESETS } from "../types";
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
import { WorkspacePanel } from "@/components/ui/workspace";
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
  onVoiceSaved?: (voiceId: string) => void;
  embeddedInModal?: boolean;
}

interface VoiceDesignCandidate {
  id: string;
  label: string;
  nuance: string;
  prompt: string;
  record: SpeechHistoryRecord;
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

export function VoiceDesignWorkspace({
  selectedModel,
  selectedModelReady = false,
  modelOptions = [],
  onSelectModel,
  onOpenModelManager,
  onModelRequired,
  onVoiceSaved,
  embeddedInModal = false,
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
  const [saveVoiceName, setSaveVoiceName] = useState("");
  const [savingVoice, setSavingVoice] = useState(false);
  const [saveVoiceStatus, setSaveVoiceStatus] = useState<{
    tone: "success" | "error";
    message: string;
  } | null>(null);

  const selectedCandidate = useMemo(
    () => candidates.find((candidate) => candidate.id === selectedCandidateId) ?? null,
    [candidates, selectedCandidateId],
  );

  useEffect(() => {
    if (!selectedCandidate) {
      setSaveVoiceName("");
      return;
    }
    setSaveVoiceName((current) =>
      current.trim() ? current : `${selectedCandidate.label} voice`,
    );
    setSaveVoiceStatus(null);
  }, [selectedCandidate?.id]);

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
        });
        setCandidates([...nextCandidates]);
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

      onVoiceSaved?.(createdVoice.id);
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
    <div
      className={clsx(
        "grid items-start gap-4",
        embeddedInModal ? "pb-1" : "pb-4 sm:pb-5",
      )}
    >
      <WorkspacePanel className="p-4">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0 flex-1">
            <div className={VOICE_ROUTE_SECTION_LABEL_CLASS}>Active Model</div>
            {modelOptions.length > 0 ? (
              <div className="mt-2">
                <RouteModelSelect
                  value={selectedModel}
                  options={modelOptions}
                  onSelect={onSelectModel}
                  className="w-full max-w-[280px]"
                />
              </div>
            ) : null}
            <StatusBadge
              tone={selectedModelReady ? "success" : "warning"}
              className="mt-2"
            >
              {selectedModelReady
                ? "Loaded and ready"
                : "Open Models and load a VoiceDesign model"}
            </StatusBadge>
          </div>
          {onOpenModelManager ? (
            <button
              onClick={onOpenModelManager}
              className="btn btn-secondary h-8 shrink-0 rounded-[var(--radius-pill)] px-3 text-xs"
            >
              <Settings2 className="h-4 w-4" />
              Models
            </button>
          ) : null}
        </div>
      </WorkspacePanel>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
        <WorkspacePanel className="p-5">
          <div className="space-y-5">
            <div>
              <div className="mb-2 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <label className={VOICE_ROUTE_SECTION_LABEL_CLASS}>
                  Voice Direction
                  <span className="ml-1 text-red-500">*</span>
                </label>
                <div className="flex items-center gap-2 sm:justify-end">
                  <Select value={language} onValueChange={setLanguage}>
                    <SelectTrigger className="w-[10.5rem]">
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

                  <button
                    onClick={() => setShowPresets((current) => !current)}
                    className="text-xs font-medium text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)]"
                  >
                    {showPresets ? "Hide presets" : "View presets"}
                  </button>
                </div>
              </div>

              {showPresets ? (
                <div className="mb-3 grid grid-cols-1 gap-2 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3 sm:grid-cols-2">
                  {VOICE_DESIGN_PRESETS.map((preset) => (
                    <button
                      key={preset.name}
                      onClick={() => {
                        setVoiceDescription(preset.description);
                        setShowPresets(false);
                      }}
                      className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-3 text-left transition-colors hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-2)]"
                    >
                      <div className={clsx(VOICE_ROUTE_TITLE_ACCENT_CLASS, "mb-1 text-sm")}>
                        {preset.name}
                      </div>
                      <div className={clsx(VOICE_ROUTE_META_COPY_CLASS, "line-clamp-2")}>
                        {preset.description}
                      </div>
                    </button>
                  ))}
                </div>
              ) : null}

              <textarea
                value={voiceDescription}
                onChange={(event) => setVoiceDescription(event.target.value)}
                placeholder="Describe the voice you want to create. Include tone, age, use case, energy, accent, or delivery style."
                rows={4}
                className="textarea w-full border-[var(--border-muted)] bg-[var(--bg-surface-0)] py-4 text-base leading-relaxed"
              />
            </div>

            <div>
              <label className={clsx(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-2 block")}>
                Preview Script
                <span className="ml-1 text-red-500">*</span>
              </label>
              <div className="relative">
                <textarea
                  value={text}
                  onChange={(event) => setText(event.target.value)}
                  placeholder="Enter the preview text you want each designed voice to speak..."
                  rows={4}
                  disabled={generating}
                  className="textarea w-full border-[var(--border-muted)] bg-[var(--bg-surface-1)] py-4 text-base leading-relaxed"
                />
                <div className="absolute bottom-3 right-3 px-1 text-[11px] font-medium text-[var(--text-muted)]">
                  {text.length} characters
                </div>
              </div>
            </div>

            {error ? (
              <StatePanel
                title="Voice design error"
                description={error}
                icon={AlertCircle}
                tone="danger"
              />
            ) : null}

            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={handleGenerate}
                disabled={generating || !selectedModelReady}
                className="btn btn-primary h-11 min-w-[180px] rounded-lg px-4 text-sm font-semibold"
              >
                {generating ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    Generate 3 options
                  </>
                )}
              </button>

              <button
                onClick={handleReset}
                className="btn btn-ghost h-11 rounded-lg border border-transparent px-4 hover:border-[var(--border-muted)]"
              >
                Reset
              </button>
            </div>

            {generationStep ? (
              <div className="text-xs font-medium text-[var(--text-muted)]">
                {generationStep}
              </div>
            ) : null}
          </div>
        </WorkspacePanel>

        <div className="space-y-4">
          <WorkspacePanel className="p-4">
            <div className={clsx(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-3")}>
              Candidate Compare
            </div>
            {candidates.length === 0 ? (
              <div className="rounded-lg border border-dashed border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-8 text-center text-sm text-[var(--text-muted)]">
                Generate options to compare designed voices.
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
                          : "border-[var(--border-muted)] bg-[var(--bg-surface-0)]/70",
                      )}
                    >
                      <div className={clsx(VOICE_ROUTE_TITLE_ACCENT_CLASS, "text-sm")}>
                        {candidate.label}
                      </div>
                      <div className={clsx(VOICE_ROUTE_META_COPY_CLASS, "mt-1")}>
                        {candidate.nuance}
                      </div>

                      <audio
                        src={api.voiceDesignRecordAudioUrl(candidate.record.id)}
                        controls
                        preload="none"
                        className="mt-3 h-10 w-full"
                      />

                      <button
                        onClick={() => setSelectedCandidateId(candidate.id)}
                        className="btn btn-secondary mt-3 h-9 w-full text-xs"
                      >
                        {isSelected ? "Selected" : "Select"}
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </WorkspacePanel>

          {selectedCandidate ? (
            <WorkspacePanel className="p-4">
              <div className={VOICE_ROUTE_PANEL_TITLE_CLASS}>Save Selected Voice</div>
              <div className={clsx(VOICE_ROUTE_META_COPY_CLASS, "mt-1")}>
                Save the selected design as a reusable voice profile.
              </div>

              <div className="mt-3">
                <label className="mb-1.5 block text-[11px] font-medium text-[var(--text-secondary)]">
                  Voice Name
                </label>
                <input
                  value={saveVoiceName}
                  onChange={(event) => setSaveVoiceName(event.target.value)}
                  placeholder="e.g. Support Voice"
                  className="input h-10 w-full border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-sm"
                  disabled={savingVoice}
                />
              </div>

              <div className="mt-3 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2 text-xs text-[var(--text-muted)] break-words">
                Selected candidate prompt: {selectedCandidate.prompt}
              </div>

              <button
                onClick={handleSaveVoice}
                disabled={savingVoice || !saveVoiceName.trim()}
                className="btn btn-secondary mt-3 h-10 w-full gap-2 border-[var(--border-muted)]"
              >
                {savingVoice ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <BookmarkPlus className="h-4 w-4" />
                    Save to My Voices
                  </>
                )}
              </button>

              {saveVoiceStatus ? (
                <div
                  className={clsx(
                    "mt-3 rounded-lg border p-3 text-xs font-medium",
                    saveVoiceStatus.tone === "success"
                      ? "border-[var(--status-positive-border)] bg-[var(--status-positive-bg)] text-[var(--status-positive-text)]"
                      : "border-[var(--danger-border)] bg-[var(--danger-bg)] text-[var(--danger-text)]",
                  )}
                >
                  <div className="flex items-center gap-2">
                    {saveVoiceStatus.tone === "success" ? (
                      <CheckCircle2 className="h-3.5 w-3.5" />
                    ) : (
                      <AlertCircle className="h-3.5 w-3.5" />
                    )}
                    {saveVoiceStatus.message}
                  </div>
                </div>
              ) : null}
            </WorkspacePanel>
          ) : null}
        </div>
      </div>
    </div>
  );
}
