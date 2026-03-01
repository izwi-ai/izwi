import { useState, useMemo } from "react";
import {
  Download,
  Play,
  Square,
  Trash2,
  HardDrive,
  Search,
  Loader2,
  X,
  RefreshCw,
} from "lucide-react";
import { ModelInfo } from "../api";
import { PageHeader, PageShell } from "../components/PageShell";
import { withQwen3Prefix } from "../utils/modelDisplay";
import clsx from "clsx";

interface MyModelsPageProps {
  models: ModelInfo[];
  loading: boolean;
  downloadProgress: Record<
    string,
    { percent: number; currentFile: string; status: string }
  >;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onRefresh: () => void;
}

type FilterType = "all" | "downloaded" | "loaded" | "not_downloaded";
type CategoryType = "all" | "tts" | "asr" | "chat";
type ProviderSection = { provider: string; models: ModelInfo[] };

const PROVIDER_ORDER = [
  "Qwen",
  "Liquid AI",
  "Google",
  "NVIDIA",
  "Mistral AI",
  "hexgrad",
  "Other",
] as const;

export const MODEL_DETAILS: Record<
  string,
  {
    shortName: string;
    fullName: string;
    description: string;
    category: "tts" | "asr" | "chat";
    capabilities: string[];
    size: string;
  }
> = {
  // TTS 0.6B Base models
  "Qwen3-TTS-12Hz-0.6B-Base": {
    shortName: "0.6B Base",
    fullName: "Qwen3-TTS 12Hz 0.6B Base",
    description: "Voice cloning with reference audio samples",
    category: "tts",
    capabilities: ["Voice Cloning"],
    size: "2.3 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-Base-4bit": {
    shortName: "0.6B Base 4-bit",
    fullName: "Qwen3-TTS 12Hz 0.6B Base (MLX 4-bit)",
    description:
      "Quantized base model for lower VRAM without losing cloning support",
    category: "tts",
    capabilities: ["Voice Cloning", "4-bit"],
    size: "1.6 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-Base-8bit": {
    shortName: "0.6B Base 8-bit",
    fullName: "Qwen3-TTS 12Hz 0.6B Base (MLX 8-bit)",
    description:
      "8-bit MLX weights for better quality while staying memory friendly",
    category: "tts",
    capabilities: ["Voice Cloning", "8-bit"],
    size: "1.9 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-Base-bf16": {
    shortName: "0.6B Base BF16",
    fullName: "Qwen3-TTS 12Hz 0.6B Base (MLX bf16)",
    description: "BF16 MLX weights for highest fidelity base voices",
    category: "tts",
    capabilities: ["Voice Cloning", "BF16"],
    size: "2.3 GB",
  },
  // TTS 0.6B CustomVoice models
  "Qwen3-TTS-12Hz-0.6B-CustomVoice": {
    shortName: "0.6B CustomVoice",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice",
    description: "Pre-trained with 9 built-in voice profiles",
    category: "tts",
    capabilities: ["Text to Speech"],
    size: "2.3 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit": {
    shortName: "0.6B Custom 4-bit",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice (MLX 4-bit)",
    description: "Quantized CustomVoice for laptops with tight memory",
    category: "tts",
    capabilities: ["Text to Speech", "4-bit"],
    size: "1.6 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit": {
    shortName: "0.6B Custom 8-bit",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice (MLX 8-bit)",
    description:
      "Balanced 8-bit CustomVoice for better quality with reduced VRAM",
    category: "tts",
    capabilities: ["Text to Speech", "8-bit"],
    size: "1.8 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16": {
    shortName: "0.6B Custom BF16",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice (MLX bf16)",
    description: "BF16 precision for premium CustomVoice rendering",
    category: "tts",
    capabilities: ["Text to Speech", "BF16"],
    size: "2.3 GB",
  },
  // TTS 1.7B Base models
  "Qwen3-TTS-12Hz-1.7B-Base": {
    shortName: "1.7B Base",
    fullName: "Qwen3-TTS 12Hz 1.7B Base",
    description: "Higher quality voice cloning capabilities",
    category: "tts",
    capabilities: ["Voice Cloning"],
    size: "4.2 GB",
  },
  "Qwen3-TTS-12Hz-1.7B-Base-4bit": {
    shortName: "1.7B Base 4-bit",
    fullName: "Qwen3-TTS 12Hz 1.7B Base (MLX 4-bit)",
    description: "Quantized base model for lower-memory voice cloning",
    category: "tts",
    capabilities: ["Voice Cloning", "4-bit"],
    size: "2.2 GB",
  },
  // TTS 1.7B CustomVoice
  "Qwen3-TTS-12Hz-1.7B-CustomVoice": {
    shortName: "1.7B CustomVoice",
    fullName: "Qwen3-TTS 12Hz 1.7B CustomVoice",
    description: "Premium quality with 9 built-in voices",
    category: "tts",
    capabilities: ["Text to Speech"],
    size: "4.2 GB",
  },
  "Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit": {
    shortName: "1.7B Custom 4-bit",
    fullName: "Qwen3-TTS 12Hz 1.7B CustomVoice (MLX 4-bit)",
    description: "Quantized CustomVoice model with lower VRAM requirements",
    category: "tts",
    capabilities: ["Text to Speech", "4-bit"],
    size: "2.2 GB",
  },
  // TTS 1.7B VoiceDesign models
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign": {
    shortName: "1.7B VoiceDesign",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign",
    description: "Generate voices from text descriptions",
    category: "tts",
    capabilities: ["Voice Design"],
    size: "4.2 GB",
  },
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit": {
    shortName: "1.7B Design 4-bit",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign (MLX 4-bit)",
    description: "Quantized VoiceDesign for creative voices on 16GB devices",
    category: "tts",
    capabilities: ["Voice Design", "4-bit"],
    size: "2.2 GB",
  },
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit": {
    shortName: "1.7B Design 8-bit",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign (MLX 8-bit)",
    description: "8-bit VoiceDesign balancing VRAM usage and quality",
    category: "tts",
    capabilities: ["Voice Design", "8-bit"],
    size: "2.9 GB",
  },
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16": {
    shortName: "1.7B Design BF16",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign (MLX bf16)",
    description: "BF16 VoiceDesign with best timbre quality",
    category: "tts",
    capabilities: ["Voice Design", "BF16"],
    size: "4.2 GB",
  },
  // LFM2 Audio
  "LFM2-Audio-1.5B": {
    shortName: "LFM2 Audio 1.5B",
    fullName: "LFM2-Audio 1.5B by Liquid AI",
    description: "End-to-end audio foundation model",
    category: "tts",
    capabilities: ["TTS", "ASR", "Audio Chat"],
    size: "3.0 GB",
  },
  "LFM2.5-Audio-1.5B": {
    shortName: "LFM2.5 Audio 1.5B",
    fullName: "LFM2.5-Audio 1.5B by Liquid AI",
    description: "Improved end-to-end audio foundation model",
    category: "tts",
    capabilities: ["TTS", "ASR", "Audio Chat"],
    size: "3.2 GB",
  },
  "LFM2.5-Audio-1.5B-4bit": {
    shortName: "LFM2.5 Audio 1.5B 4-bit",
    fullName: "LFM2.5-Audio 1.5B (MLX 4-bit)",
    description: "Quantized LFM2.5 audio model for lower memory usage",
    category: "tts",
    capabilities: ["TTS", "ASR", "Audio Chat", "4-bit"],
    size: "0.8 GB",
  },
  "Kokoro-82M": {
    shortName: "Kokoro 82M",
    fullName: "Kokoro-82M by hexgrad",
    description:
      "Lightweight multilingual TTS model with 54 built-in voices (Rust runtime integration pending)",
    category: "tts",
    capabilities: ["Text to Speech", "54 Voices", "24 kHz"],
    size: "346 MB",
  },
  // Text Chat
  "Qwen3-0.6B": {
    shortName: "Qwen3 Chat 0.6B",
    fullName: "Qwen3 0.6B",
    description: "Compact text-to-text model for local chat",
    category: "chat",
    capabilities: ["Text Chat", "Full precision"],
    size: "1.4 GB",
  },
  "Qwen3-0.6B-4bit": {
    shortName: "Qwen3 Chat 0.6B",
    fullName: "Qwen3 0.6B (MLX 4-bit)",
    description: "Compact text-to-text model for local chat",
    category: "chat",
    capabilities: ["Text Chat", "4-bit"],
    size: "0.9 GB",
  },
  "Qwen3-1.7B-4bit": {
    shortName: "Qwen3 Chat 1.7B",
    fullName: "Qwen3 1.7B (MLX 4-bit)",
    description: "Higher-quality 1.7B chat model in 4-bit precision",
    category: "chat",
    capabilities: ["Text Chat", "4-bit"],
    size: "1.1 GB",
  },
  "Qwen3-0.6B-GGUF": {
    shortName: "Qwen3 Chat 0.6B GGUF",
    fullName: "Qwen3 0.6B (GGUF Q8_0)",
    description: "Compact text-to-text model in GGUF format",
    category: "chat",
    capabilities: ["Text Chat", "GGUF", "Q8_0"],
    size: "1.0 GB",
  },
  "Qwen3-1.7B-GGUF": {
    shortName: "Qwen3 Chat 1.7B GGUF",
    fullName: "Qwen3 1.7B (GGUF Q8_0)",
    description: "Higher-quality 1.7B chat model in GGUF format",
    category: "chat",
    capabilities: ["Text Chat", "GGUF", "Q8_0"],
    size: "2.2 GB",
  },
  "Qwen3-4B-GGUF": {
    shortName: "Qwen3 Chat 4B GGUF",
    fullName: "Qwen3 4B (GGUF Q4_K_M)",
    description: "4B chat model in GGUF format (Q4_K_M only)",
    category: "chat",
    capabilities: ["Text Chat", "GGUF", "Q4_K_M"],
    size: "2.5 GB",
  },
  "Qwen3-8B-GGUF": {
    shortName: "Qwen3 Chat 8B GGUF",
    fullName: "Qwen3 8B (GGUF Q4_K_M)",
    description: "8B chat model in GGUF format (Q4_K_M only)",
    category: "chat",
    capabilities: ["Text Chat", "GGUF", "Q4_K_M"],
    size: "5.2 GB",
  },
  "Qwen3-14B-GGUF": {
    shortName: "Qwen3 Chat 14B GGUF",
    fullName: "Qwen3 14B (GGUF Q4_K_M)",
    description: "14B chat model in GGUF format (Q4_K_M only)",
    category: "chat",
    capabilities: ["Text Chat", "GGUF", "Q4_K_M"],
    size: "9.2 GB",
  },
  "Gemma-3-1b-it": {
    shortName: "Gemma 3 1B",
    fullName: "Gemma 3 1B Instruct",
    description: "Lightweight Gemma 3 instruction model for local chat",
    category: "chat",
    capabilities: ["Text Chat", "Instruction Tuned"],
    size: "2.1 GB",
  },
  "Gemma-3-4b-it": {
    shortName: "Gemma 3 4B",
    fullName: "Gemma 3 4B Instruct",
    description: "Higher-quality Gemma 3 instruction model for local chat",
    category: "chat",
    capabilities: ["Text Chat", "Instruction Tuned"],
    size: "8.0 GB",
  },
  // ASR 0.6B models
  "Qwen3-ASR-0.6B": {
    shortName: "ASR 0.6B",
    fullName: "Qwen3-ASR 0.6B",
    description: "Fast speech-to-text, 52 languages",
    category: "asr",
    capabilities: ["Transcription"],
    size: "1.8 GB",
  },
  "Qwen3-ASR-0.6B-4bit": {
    shortName: "ASR 0.6B 4-bit",
    fullName: "Qwen3-ASR 0.6B (MLX 4-bit)",
    description: "Lightweight ASR for real-time transcription on smaller GPUs",
    category: "asr",
    capabilities: ["Transcription", "4-bit"],
    size: "0.7 GB",
  },
  "Qwen3-ASR-0.6B-8bit": {
    shortName: "ASR 0.6B 8-bit",
    fullName: "Qwen3-ASR 0.6B (MLX 8-bit)",
    description: "8-bit ASR with higher accuracy and modest footprint",
    category: "asr",
    capabilities: ["Transcription", "8-bit"],
    size: "0.9 GB",
  },
  "Qwen3-ASR-0.6B-bf16": {
    shortName: "ASR 0.6B BF16",
    fullName: "Qwen3-ASR 0.6B (MLX bf16)",
    description: "BF16 precision ASR for top accuracy",
    category: "asr",
    capabilities: ["Transcription", "BF16"],
    size: "1.5 GB",
  },
  // ASR 1.7B models
  "Qwen3-ASR-1.7B": {
    shortName: "ASR 1.7B",
    fullName: "Qwen3-ASR 1.7B",
    description: "High-quality speech-to-text, 52 languages",
    category: "asr",
    capabilities: ["Transcription"],
    size: "4.4 GB",
  },
  "Qwen3-ASR-1.7B-4bit": {
    shortName: "ASR 1.7B 4-bit",
    fullName: "Qwen3-ASR 1.7B (MLX 4-bit)",
    description: "Quantized 1.7B ASR for RTX 4090 / M3 workloads",
    category: "asr",
    capabilities: ["Transcription", "4-bit"],
    size: "1.5 GB",
  },
  "Qwen3-ASR-1.7B-8bit": {
    shortName: "ASR 1.7B 8-bit",
    fullName: "Qwen3-ASR 1.7B (MLX 8-bit)",
    description: "8-bit ASR for high fidelity transcripts on Apple Silicon",
    category: "asr",
    capabilities: ["Transcription", "8-bit"],
    size: "2.3 GB",
  },
  "Qwen3-ASR-1.7B-bf16": {
    shortName: "ASR 1.7B BF16",
    fullName: "Qwen3-ASR 1.7B (MLX bf16)",
    description: "BF16 ASR providing maximum quality and accuracy",
    category: "asr",
    capabilities: ["Transcription", "BF16"],
    size: "3.8 GB",
  },
  "Qwen3-ForcedAligner-0.6B": {
    shortName: "ForcedAligner 0.6B",
    fullName: "Qwen3-ForcedAligner 0.6B",
    description: "Aligns transcript text to precise speech timestamps",
    category: "asr",
    capabilities: ["Forced Alignment", "Word timestamps"],
    size: "1.7 GB",
  },
  "Qwen3-ForcedAligner-0.6B-4bit": {
    shortName: "ForcedAligner 0.6B 4-bit",
    fullName: "Qwen3-ForcedAligner 0.6B (MLX 4-bit)",
    description: "Quantized forced aligner for low-memory alignment workflows",
    category: "asr",
    capabilities: ["Forced Alignment", "Word timestamps", "4-bit"],
    size: "0.7 GB",
  },
  "Parakeet-TDT-0.6B-v2": {
    shortName: "Parakeet v2",
    fullName: "Parakeet-TDT 0.6B v2",
    description: "English FastConformer-TDT ASR model in .nemo format",
    category: "asr",
    capabilities: ["Transcription", "Word timestamps"],
    size: "4.6 GB",
  },
  "Parakeet-TDT-0.6B-v3": {
    shortName: "Parakeet v3",
    fullName: "Parakeet-TDT 0.6B v3",
    description: "Multilingual FastConformer-TDT ASR model in .nemo format",
    category: "asr",
    capabilities: ["Transcription", "25 EU languages"],
    size: "9.3 GB",
  },
  "diar_streaming_sortformer_4spk-v2.1": {
    shortName: "Sortformer 4spk",
    fullName: "Streaming Sortformer 4spk v2.1",
    description:
      "Streaming speaker diarization model from NVIDIA in .nemo format",
    category: "asr",
    capabilities: ["Diarization", "Up to 4 speakers", "Streaming"],
    size: "0.5 GB",
  },
  // Voxtral
  "Voxtral-Mini-4B-Realtime-2602": {
    shortName: "Voxtral 4B",
    fullName: "Voxtral Mini 4B Realtime",
    description: "Realtime streaming ASR from Mistral AI",
    category: "asr",
    capabilities: ["Transcription", "Realtime"],
    size: "8.0 GB",
  },
};

function parseSize(sizeStr: string): number {
  const match = sizeStr.match(/^([\d.]+)\s*(GB|MB|KB|B)?$/i);
  if (!match) return 0;
  const value = parseFloat(match[1]);
  const unit = (match[2] || "B").toUpperCase();
  const multipliers: Record<string, number> = {
    B: 1,
    KB: 1024,
    MB: 1024 * 1024,
    GB: 1024 * 1024 * 1024,
  };
  return value * (multipliers[unit] || 1);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function getModelSizeBytes(model: ModelInfo): number {
  if (model.size_bytes !== null && model.size_bytes > 0) {
    return model.size_bytes;
  }
  const fallback = MODEL_DETAILS[model.variant]?.size;
  return fallback ? parseSize(fallback) : 0;
}

function getModelSizeLabel(model: ModelInfo): string {
  const bytes = getModelSizeBytes(model);
  return bytes > 0 ? formatBytes(bytes) : "Size unknown";
}

function getStatusDotClass(status: ModelInfo["status"]): string {
  switch (status) {
    case "ready":
      return "bg-[var(--status-positive-solid)]";
    case "downloaded":
      return "bg-[var(--text-secondary)]";
    case "downloading":
    case "loading":
      return "bg-[var(--status-warning-text)]";
    case "error":
      return "bg-[var(--danger-text)]";
    default:
      return "bg-[var(--text-subtle)]";
  }
}

function getProviderLabel(variant: string): string {
  if (variant.startsWith("Qwen3-")) return "Qwen";
  if (variant.startsWith("LFM2")) return "Liquid AI";
  if (variant.startsWith("Gemma-")) return "Google";
  if (
    variant.startsWith("Parakeet-") ||
    variant.startsWith("diar_streaming_sortformer")
  ) {
    return "NVIDIA";
  }
  if (variant.startsWith("Voxtral-")) return "Mistral AI";
  if (variant.startsWith("Kokoro-")) return "hexgrad";
  return "Other";
}

function compareProviders(left: string, right: string): number {
  const leftRank = PROVIDER_ORDER.indexOf(left as (typeof PROVIDER_ORDER)[number]);
  const rightRank = PROVIDER_ORDER.indexOf(
    right as (typeof PROVIDER_ORDER)[number],
  );
  const normalizedLeftRank = leftRank === -1 ? Number.MAX_SAFE_INTEGER : leftRank;
  const normalizedRightRank =
    rightRank === -1 ? Number.MAX_SAFE_INTEGER : rightRank;
  if (normalizedLeftRank !== normalizedRightRank) {
    return normalizedLeftRank - normalizedRightRank;
  }
  return left.localeCompare(right);
}

function requiresManualDownload(variant: string): boolean {
  return variant === "Gemma-3-1b-it";
}

export function MyModelsPage({
  models,
  loading,
  downloadProgress,
  onDownload,
  onCancelDownload,
  onLoad,
  onUnload,
  onDelete,
  onRefresh,
}: MyModelsPageProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<FilterType>("all");
  const [categoryFilter, setCategoryFilter] = useState<CategoryType>("all");
  const [deleteModalVariant, setDeleteModalVariant] = useState<string | null>(
    null,
  );
  const [isRefreshing, setIsRefreshing] = useState(false);

  const filteredModels = useMemo(() => {
    return models
      .filter((m) => !m.variant.includes("Tokenizer"))
      .filter((m) => {
        const details = MODEL_DETAILS[m.variant];
        if (!details) return false;

        // Search filter
        if (searchQuery) {
          const query = searchQuery.toLowerCase();
          const providerLabel = getProviderLabel(m.variant).toLowerCase();
          const matchesSearch =
            details.shortName.toLowerCase().includes(query) ||
            details.fullName.toLowerCase().includes(query) ||
            details.description.toLowerCase().includes(query) ||
            details.capabilities.some((c) => c.toLowerCase().includes(query)) ||
            providerLabel.includes(query);
          if (!matchesSearch) return false;
        }

        // Status filter
        if (statusFilter !== "all") {
          if (statusFilter === "downloaded" && m.status !== "downloaded")
            return false;
          if (statusFilter === "loaded" && m.status !== "ready") return false;
          if (
            statusFilter === "not_downloaded" &&
            m.status !== "not_downloaded"
          )
            return false;
        }

        // Category filter
        if (categoryFilter !== "all" && details.category !== categoryFilter) {
          return false;
        }

        return true;
      })
      .sort((a, b) => {
        // Stable sort independent of status so cards do not jump while downloading/loading.
        const sizeA = getModelSizeBytes(a);
        const sizeB = getModelSizeBytes(b);
        if (sizeA !== sizeB) return sizeA - sizeB;
        return a.variant.localeCompare(b.variant);
      });
  }, [models, searchQuery, statusFilter, categoryFilter]);

  const providerSections = useMemo<ProviderSection[]>(() => {
    const grouped = new Map<string, ModelInfo[]>();
    for (const model of filteredModels) {
      const provider = getProviderLabel(model.variant);
      const bucket = grouped.get(provider);
      if (bucket) {
        bucket.push(model);
      } else {
        grouped.set(provider, [model]);
      }
    }

    return Array.from(grouped.entries())
      .sort(([left], [right]) => compareProviders(left, right))
      .map(([provider, groupedModels]) => ({
        provider,
        models: groupedModels,
      }));
  }, [filteredModels]);

  const stats = useMemo(() => {
    const visibleModels = models.filter(
      (m) => !m.variant.includes("Tokenizer") && MODEL_DETAILS[m.variant],
    );
    return {
      total: visibleModels.length,
      loaded: visibleModels.filter((m) => m.status === "ready").length,
      downloaded: visibleModels.filter(
        (m) => m.status === "downloaded" || m.status === "ready",
      ).length,
      totalSize: visibleModels
        .filter((m) => m.status === "downloaded" || m.status === "ready")
        .reduce((acc, m) => acc + getModelSizeBytes(m), 0),
    };
  }, [models]);

  const deleteTargetDetails =
    deleteModalVariant && MODEL_DETAILS[deleteModalVariant]
      ? MODEL_DETAILS[deleteModalVariant]
      : null;
  const deleteTargetName =
    deleteTargetDetails && deleteModalVariant
      ? withQwen3Prefix(deleteTargetDetails.shortName, deleteModalVariant)
      : deleteModalVariant;

  if (loading) {
    return (
      <PageShell>
        <div className="flex items-center justify-center gap-2 py-24 text-[var(--text-muted)]">
          <Loader2 className="w-4 h-4 animate-spin" />
          <p className="text-sm">Loading models...</p>
        </div>
      </PageShell>
    );
  }

  const hasActiveFilters =
    searchQuery.trim().length > 0 ||
    statusFilter !== "all" ||
    categoryFilter !== "all";

  return (
    <PageShell className="space-y-4">
      <PageHeader
        title="Models"
        description="Download, load, and remove local models."
        actions={
          <div className="flex flex-wrap items-center gap-2">
            {onRefresh && (
              <button
                onClick={async () => {
                  setIsRefreshing(true);
                  await onRefresh();
                  setIsRefreshing(false);
                }}
                disabled={isRefreshing}
                className="flex items-center gap-1.5 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2 text-xs font-medium text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-surface-2)] disabled:opacity-60"
                title="Refresh models"
              >
                <RefreshCw
                  className={clsx(
                    "w-3.5 h-3.5",
                    isRefreshing && "animate-spin",
                  )}
                />
                Refresh
              </button>
            )}

            <div className="flex items-center gap-2 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2">
              <HardDrive className="w-4 h-4 text-[var(--text-subtle)]" />
              <div className="text-sm">
                <span className="font-medium text-[var(--text-primary)]">
                  {formatBytes(stats.totalSize)}
                </span>
                <span className="ml-1 text-[var(--text-muted)]">used</span>
              </div>
            </div>

            <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2 text-sm">
              <span className="font-medium text-[var(--text-primary)]">
                {stats.loaded}
              </span>
              <span className="ml-1 text-[var(--text-muted)]">loaded</span>
              <span className="mx-1 text-[var(--text-subtle)]">/</span>
              <span className="text-[var(--text-muted)]">
                {stats.downloaded}
              </span>
              <span className="ml-1 text-[var(--text-subtle)]">downloaded</span>
            </div>
          </div>
        }
      />

      <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3 sm:p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--text-subtle)]" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search models..."
              className="w-full rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-2)] py-2.5 pl-10 pr-3 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-subtle)] focus:border-[var(--border-strong)] focus:outline-none"
            />
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            {[
              { id: "all" as FilterType, label: "All" },
              { id: "loaded" as FilterType, label: "Loaded" },
              { id: "downloaded" as FilterType, label: "Downloaded" },
              { id: "not_downloaded" as FilterType, label: "Not downloaded" },
            ].map((option) => (
              <button
                key={option.id}
                onClick={() => setStatusFilter(option.id)}
                className={clsx(
                  "rounded-md border px-3 py-1.5 text-xs font-medium transition-colors",
                  statusFilter === option.id
                    ? "border-[var(--border-strong)] bg-[var(--bg-surface-3)] text-[var(--text-primary)]"
                    : "border-[var(--border-muted)] bg-[var(--bg-surface-1)] text-[var(--text-muted)] hover:text-[var(--text-primary)]",
                )}
              >
                {option.label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <select
              value={categoryFilter}
              onChange={(event) =>
                setCategoryFilter(event.target.value as CategoryType)
              }
              className="rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-secondary)] focus:border-[var(--border-strong)] focus:outline-none"
              aria-label="Filter models by category"
            >
              <option value="all">All categories</option>
              <option value="tts">Text to Speech</option>
              <option value="asr">Transcription</option>
              <option value="chat">Chat</option>
            </select>

            {hasActiveFilters && (
              <button
                onClick={() => {
                  setSearchQuery("");
                  setStatusFilter("all");
                  setCategoryFilter("all");
                }}
                className="flex items-center gap-1 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-1.5 text-xs text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)]"
              >
                <X className="h-3.5 w-3.5" />
                Clear
              </button>
            )}
          </div>
        </div>
      </div>

      {filteredModels.length === 0 ? (
        <div className="flex flex-col items-center justify-center rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] py-16 text-center">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-2)]">
            <HardDrive className="h-5 w-5 text-[var(--text-subtle)]" />
          </div>
          <h3 className="text-base font-medium text-[var(--text-primary)]">
            No models found
          </h3>
          <p className="mt-1 max-w-sm text-sm text-[var(--text-muted)]">
            {hasActiveFilters
              ? "Try adjusting your filters."
              : "Download a model to get started."}
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {providerSections.map((section) => (
            <section key={section.provider} className="space-y-2">
              <div className="flex items-center gap-2 px-1">
                <h3 className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-subtle)]">
                  {section.provider}
                </h3>
                <span className="text-[10px] text-[var(--text-subtle)]">
                  {section.models.length}
                </span>
                <div className="h-px flex-1 bg-[var(--border-muted)]" />
              </div>

              <div className="space-y-2">
                {section.models.map((model) => {
                  const details = MODEL_DETAILS[model.variant];
                  if (!details) return null;

                  const displayName = withQwen3Prefix(
                    details.shortName,
                    model.variant,
                  );
                  const isDownloading = model.status === "downloading";
                  const isLoading = model.status === "loading";
                  const isReady = model.status === "ready";
                  const isDownloaded = model.status === "downloaded";
                  const progressValue = downloadProgress[model.variant];
                  const progress =
                    progressValue?.percent ?? model.download_progress ?? 0;

                  return (
                    <div
                      key={model.variant}
                      className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2.5"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div className="min-w-0 flex items-center gap-2">
                          {isDownloading || isLoading ? (
                            <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-[var(--status-warning-text)]" />
                          ) : (
                            <span
                              className={clsx(
                                "h-2 w-2 shrink-0 rounded-full",
                                getStatusDotClass(model.status),
                              )}
                            />
                          )}
                          <h3 className="truncate text-sm font-medium text-[var(--text-primary)]">
                            {displayName}
                          </h3>
                        </div>

                        <div className="shrink-0 flex items-center gap-1.5">
                          <span className="mr-1 text-xs text-[var(--text-subtle)] whitespace-nowrap">
                            {getModelSizeLabel(model)}
                          </span>
                          {isDownloading && (
                            <span className="text-xs text-[var(--status-warning-text)] whitespace-nowrap">
                              {Math.round(progress)}%
                            </span>
                          )}
                          {model.status === "not_downloaded" &&
                            (requiresManualDownload(model.variant) ? (
                              <button
                                className="flex items-center gap-1.5 rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-muted)] disabled:cursor-not-allowed disabled:opacity-60"
                                disabled
                                title="Manual download required. See docs/user/manual-gemma-3-1b-download.md."
                              >
                                <Download className="h-3.5 w-3.5" />
                                Manual download
                              </button>
                            ) : (
                              <button
                                onClick={() => onDownload(model.variant)}
                                className="flex items-center gap-1.5 rounded-md bg-[var(--accent-solid)] px-3 py-1.5 text-xs font-medium text-[var(--text-on-accent)] transition-opacity hover:opacity-90"
                              >
                                <Download className="h-3.5 w-3.5" />
                                Download
                              </button>
                            ))}

                          {isDownloading && onCancelDownload && (
                            <button
                              onClick={() => onCancelDownload(model.variant)}
                              className="flex items-center gap-1 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-2.5 py-1.5 text-xs font-medium text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]"
                            >
                              <X className="h-3.5 w-3.5" />
                              Cancel
                            </button>
                          )}

                          {isDownloaded && (
                            <>
                              <button
                                onClick={() => onLoad(model.variant)}
                                className="flex items-center gap-1.5 rounded-md bg-[var(--accent-solid)] px-3 py-1.5 text-xs font-medium text-[var(--text-on-accent)] transition-opacity hover:opacity-90"
                              >
                                <Play className="h-3.5 w-3.5" />
                                Load
                              </button>
                              <button
                                onClick={() => setDeleteModalVariant(model.variant)}
                                className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]"
                                title="Delete model"
                                aria-label={`Delete ${displayName}`}
                              >
                                <Trash2 className="h-3.5 w-3.5" />
                              </button>
                            </>
                          )}

                          {isReady && (
                            <>
                              <button
                                onClick={() => onUnload(model.variant)}
                                className="flex items-center gap-1.5 rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-surface-3)]"
                              >
                                <Square className="h-3.5 w-3.5" />
                                Unload
                              </button>
                              <button
                                onClick={() => setDeleteModalVariant(model.variant)}
                                className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]"
                                title="Delete model"
                                aria-label={`Delete ${displayName}`}
                              >
                                <Trash2 className="h-3.5 w-3.5" />
                              </button>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>
          ))}
        </div>
      )}

      {deleteModalVariant && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          onClick={() => setDeleteModalVariant(null)}
        >
          <div
            className="w-full max-w-md rounded-xl border border-[var(--danger-border)] bg-[var(--bg-surface-1)] p-5 shadow-xl"
            onClick={(event) => event.stopPropagation()}
          >
            <h3 className="text-sm font-semibold text-[var(--text-primary)]">
              Delete model?
            </h3>
            <p className="mt-2 text-sm text-[var(--text-muted)]">
              This will remove the downloaded model files from local storage.
            </p>
            <p className="mt-3 truncate text-xs text-[var(--text-subtle)]">
              {deleteTargetName || deleteModalVariant}
            </p>

            <div className="mt-5 flex items-center justify-end gap-2">
              <button
                onClick={() => setDeleteModalVariant(null)}
                className="rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-surface-3)]"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  onDelete(deleteModalVariant);
                  setDeleteModalVariant(null);
                }}
                className="inline-flex items-center gap-1.5 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-1.5 text-xs font-medium text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]"
              >
                <Trash2 className="h-3.5 w-3.5" />
                Delete model
              </button>
            </div>
          </div>
        </div>
      )}
    </PageShell>
  );
}
