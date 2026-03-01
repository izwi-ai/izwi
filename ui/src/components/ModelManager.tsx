import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Check,
  Download,
  Play,
  Square,
  Trash2,
  ChevronRight,
  Loader2,
  X,
} from "lucide-react";
import { ModelInfo } from "../api";
import { withQwen3Prefix } from "../utils/modelDisplay";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

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

function requiresManualDownload(variant: string): boolean {
  return variant === "Gemma-3-1b-it";
}

interface ModelManagerProps {
  models: ModelInfo[];
  selectedModel: string | null;
  onDownload: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onSelect: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  downloadProgress: Record<
    string,
    {
      percent: number;
      currentFile: string;
      status: string;
      downloadedBytes: number;
      totalBytes: number;
    }
  >;
  modelFilter?: (variant: string) => boolean;
  isModelDisabled?: (variant: string) => boolean;
  disabledModelLabel?: string;
  emptyStateTitle?: string;
  emptyStateDescription?: string;
}

const MODEL_DETAILS: Record<
  string,
  {
    shortName: string;
    fullName: string;
    description: string;
    features: string[];
    size: string;
    quantization?: string;
  }
> = {
  "Qwen3-TTS-12Hz-0.6B-Base": {
    shortName: "0.6B Base",
    fullName: "Qwen3-TTS 12Hz 0.6B Base Model",
    description: "Voice cloning with reference audio samples",
    features: ["Voice cloning", "Reference audio required", "Fast inference"],
    size: "2.3 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-Base-4bit": {
    shortName: "0.6B Base 4-bit",
    fullName: "Qwen3-TTS 12Hz 0.6B Base (MLX 4-bit)",
    description:
      "Quantized base model for lower VRAM without losing cloning support",
    features: ["Voice cloning", "Reference audio", "4-bit quantized"],
    size: "1.6 GB",
    quantization: "4-bit",
  },
  "Qwen3-TTS-12Hz-0.6B-Base-8bit": {
    shortName: "0.6B Base 8-bit",
    fullName: "Qwen3-TTS 12Hz 0.6B Base (MLX 8-bit)",
    description:
      "8-bit MLX weights for better quality while staying memory friendly",
    features: ["Voice cloning", "Reference audio", "8-bit quantized"],
    size: "1.9 GB",
    quantization: "8-bit",
  },
  "Qwen3-TTS-12Hz-0.6B-Base-bf16": {
    shortName: "0.6B Base BF16",
    fullName: "Qwen3-TTS 12Hz 0.6B Base (MLX bf16)",
    description: "BF16 MLX weights for highest fidelity base voices",
    features: ["Voice cloning", "Reference audio", "bf16 weights"],
    size: "2.3 GB",
    quantization: "bf16",
  },
  "Qwen3-TTS-12Hz-0.6B-CustomVoice": {
    shortName: "0.6B Custom",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice Model",
    description: "Pre-trained with 9 built-in voice profiles",
    features: ["9 built-in voices", "No reference needed", "Fast generation"],
    size: "2.3 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit": {
    shortName: "0.6B Custom 4-bit",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice (MLX 4-bit)",
    description: "Quantized CustomVoice for laptops with tight memory",
    features: ["9 built-in voices", "4-bit quantized", "Fast inference"],
    size: "1.6 GB",
    quantization: "4-bit",
  },
  "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit": {
    shortName: "0.6B Custom 8-bit",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice (MLX 8-bit)",
    description:
      "Balanced 8-bit CustomVoice for better quality with reduced VRAM",
    features: ["9 built-in voices", "8-bit quantized", "Balanced quality"],
    size: "1.8 GB",
    quantization: "8-bit",
  },
  "Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16": {
    shortName: "0.6B Custom BF16",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice (MLX bf16)",
    description: "BF16 precision for premium CustomVoice rendering",
    features: ["9 built-in voices", "bf16 weights", "Best fidelity"],
    size: "2.3 GB",
    quantization: "bf16",
  },
  "Qwen3-TTS-12Hz-1.7B-Base": {
    shortName: "1.7B Base",
    fullName: "Qwen3-TTS 12Hz 1.7B Base Model",
    description: "Higher quality voice cloning capabilities",
    features: [
      "Superior voice cloning",
      "Reference audio required",
      "Best quality",
    ],
    size: "4.2 GB",
  },
  "Qwen3-TTS-12Hz-1.7B-Base-4bit": {
    shortName: "1.7B Base 4-bit",
    fullName: "Qwen3-TTS 12Hz 1.7B Base (MLX 4-bit)",
    description: "Quantized 1.7B Base model for lower memory voice cloning",
    features: ["Voice cloning", "Reference audio", "4-bit quantized"],
    size: "2.2 GB",
    quantization: "4-bit",
  },
  "Qwen3-TTS-12Hz-1.7B-CustomVoice": {
    shortName: "1.7B Custom",
    fullName: "Qwen3-TTS 12Hz 1.7B CustomVoice Model",
    description: "Premium quality with 9 built-in voices",
    features: ["9 built-in voices", "Highest quality", "Natural prosody"],
    size: "4.2 GB",
  },
  "Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit": {
    shortName: "1.7B Custom 4-bit",
    fullName: "Qwen3-TTS 12Hz 1.7B CustomVoice (MLX 4-bit)",
    description:
      "Quantized 1.7B CustomVoice for lower VRAM with premium voice quality",
    features: ["9 built-in voices", "4-bit quantized", "Balanced quality"],
    size: "2.2 GB",
    quantization: "4-bit",
  },
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit": {
    shortName: "1.7B Design 4-bit",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign (MLX 4-bit)",
    description: "Quantized VoiceDesign for creative voices on 16GB devices",
    features: ["Text-to-voice", "Creative control", "4-bit quantized"],
    size: "2.2 GB",
    quantization: "4-bit",
  },
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit": {
    shortName: "1.7B Design 8-bit",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign (MLX 8-bit)",
    description: "8-bit VoiceDesign balancing VRAM usage and quality",
    features: ["Text-to-voice", "Creative control", "8-bit quantized"],
    size: "2.9 GB",
    quantization: "8-bit",
  },
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16": {
    shortName: "1.7B Design BF16",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign (MLX bf16)",
    description: "BF16 VoiceDesign with best timbre quality",
    features: ["Text-to-voice", "Creative control", "bf16 weights"],
    size: "4.2 GB",
    quantization: "bf16",
  },
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign": {
    shortName: "1.7B Design",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign Model",
    description: "Generate voices from text descriptions",
    features: ["Text-to-voice", "Creative control", "Unique voices"],
    size: "4.2 GB",
  },
  "LFM2-Audio-1.5B": {
    shortName: "LFM2 1.5B",
    fullName: "LFM2-Audio 1.5B by Liquid AI",
    description:
      "End-to-end audio foundation model for TTS, ASR, and audio chat",
    features: ["TTS", "ASR", "Audio-to-audio chat", "4 voice styles"],
    size: "3.0 GB",
  },
  "LFM2.5-Audio-1.5B": {
    shortName: "LFM2.5 1.5B",
    fullName: "LFM2.5-Audio 1.5B by Liquid AI",
    description:
      "Improved LFM2 audio model for TTS, ASR, and audio-to-audio generation",
    features: ["TTS", "ASR", "Audio-to-audio chat", "4 voice styles"],
    size: "3.2 GB",
  },
  "LFM2.5-Audio-1.5B-4bit": {
    shortName: "LFM2.5 1.5B 4-bit",
    fullName: "LFM2.5-Audio 1.5B (MLX 4-bit)",
    description: "Quantized LFM2.5 audio model for lower memory usage",
    features: ["TTS", "ASR", "Audio-to-audio chat", "4-bit quantized"],
    size: "0.8 GB",
    quantization: "4-bit",
  },
  "Kokoro-82M": {
    shortName: "Kokoro 82M",
    fullName: "Kokoro-82M by hexgrad",
    description:
      "Lightweight multilingual TTS model with 54 built-in voices (Rust runtime integration pending)",
    features: [
      "TTS",
      "54 built-in voices",
      "24 kHz output",
      "PyTorch checkpoint",
    ],
    size: "346 MB",
  },
  "Qwen3-0.6B": {
    shortName: "Qwen3 Chat 0.6B",
    fullName: "Qwen3 0.6B",
    description: "Compact text-to-text chat model for local inference",
    features: ["Text chat", "Full precision", "Fast response"],
    size: "1.4 GB",
  },
  "Qwen3-0.6B-4bit": {
    shortName: "Qwen3 Chat 0.6B",
    fullName: "Qwen3 0.6B (MLX 4-bit)",
    description:
      "Compact text-to-text chat model optimized for local inference",
    features: ["Text chat", "4-bit quantized", "Fast response"],
    size: "0.9 GB",
    quantization: "4-bit",
  },
  "Qwen3-1.7B-4bit": {
    shortName: "Qwen3 Chat 1.7B",
    fullName: "Qwen3 1.7B (MLX 4-bit)",
    description: "Higher-quality 1.7B local chat model with 4-bit quantization",
    features: ["Text chat", "4-bit quantized", "Improved quality"],
    size: "1.1 GB",
    quantization: "4-bit",
  },
  "Qwen3-0.6B-GGUF": {
    shortName: "Qwen3 Chat 0.6B GGUF",
    fullName: "Qwen3 0.6B (GGUF Q8_0)",
    description: "Compact Qwen3 chat model in GGUF format",
    features: ["Text chat", "GGUF", "Q8_0 quantized"],
    size: "1.0 GB",
    quantization: "Q8_0",
  },
  "Qwen3-1.7B-GGUF": {
    shortName: "Qwen3 Chat 1.7B GGUF",
    fullName: "Qwen3 1.7B (GGUF Q8_0)",
    description: "Higher-capacity Qwen3 chat model in GGUF format",
    features: ["Text chat", "GGUF", "Q8_0 quantized"],
    size: "2.2 GB",
    quantization: "Q8_0",
  },
  "Qwen3-4B-GGUF": {
    shortName: "Qwen3 Chat 4B GGUF",
    fullName: "Qwen3 4B (GGUF Q4_K_M)",
    description:
      "Qwen3 4B chat model in GGUF format (only the Q4_K_M quantization is bundled)",
    features: ["Text chat", "GGUF", "Q4_K_M quantized"],
    size: "2.5 GB",
    quantization: "Q4_K_M",
  },
  "Qwen3-8B-GGUF": {
    shortName: "Qwen3 Chat 8B GGUF",
    fullName: "Qwen3 8B (GGUF Q4_K_M)",
    description:
      "Qwen3 8B chat model in GGUF format (only the Q4_K_M quantization is bundled)",
    features: ["Text chat", "GGUF", "Q4_K_M quantized"],
    size: "5.2 GB",
    quantization: "Q4_K_M",
  },
  "Qwen3-14B-GGUF": {
    shortName: "Qwen3 Chat 14B GGUF",
    fullName: "Qwen3 14B (GGUF Q4_K_M)",
    description:
      "Qwen3 14B chat model in GGUF format (only the Q4_K_M quantization is bundled)",
    features: ["Text chat", "GGUF", "Q4_K_M quantized"],
    size: "9.2 GB",
    quantization: "Q4_K_M",
  },
  "Gemma-3-1b-it": {
    shortName: "Gemma 3 1B",
    fullName: "Gemma 3 1B Instruct",
    description:
      "Lightweight Gemma 3 chat model for local instruction-following",
    features: ["Text chat", "Instruction tuned", "Fast responses"],
    size: "2.1 GB",
  },
  "Gemma-3-4b-it": {
    shortName: "Gemma 3 4B",
    fullName: "Gemma 3 4B Instruct",
    description: "Higher-quality Gemma 3 chat model with stronger reasoning",
    features: ["Text chat", "Instruction tuned", "Higher quality"],
    size: "8.0 GB",
  },
  "Qwen3-ASR-0.6B": {
    shortName: "ASR 0.6B",
    fullName: "Qwen3-ASR 0.6B",
    description: "Fast speech-to-text model supporting 52 languages",
    features: ["52 languages", "Language detection", "Fast inference"],
    size: "1.8 GB",
  },
  "Qwen3-ASR-0.6B-4bit": {
    shortName: "ASR 0.6B 4-bit",
    fullName: "Qwen3-ASR 0.6B (MLX 4-bit)",
    description: "Lightweight ASR for real-time transcription on smaller GPUs",
    features: ["52 languages", "4-bit quantized", "Fast inference"],
    size: "0.7 GB",
    quantization: "4-bit",
  },
  "Qwen3-ASR-0.6B-8bit": {
    shortName: "ASR 0.6B 8-bit",
    fullName: "Qwen3-ASR 0.6B (MLX 8-bit)",
    description: "8-bit ASR with higher accuracy and modest footprint",
    features: ["52 languages", "8-bit quantized", "Balanced quality"],
    size: "0.9 GB",
    quantization: "8-bit",
  },
  "Qwen3-ASR-0.6B-bf16": {
    shortName: "ASR 0.6B BF16",
    fullName: "Qwen3-ASR 0.6B (MLX bf16)",
    description: "BF16 precision ASR for top accuracy",
    features: ["52 languages", "bf16 weights", "Highest fidelity"],
    size: "1.5 GB",
    quantization: "bf16",
  },
  "Qwen3-ASR-1.7B": {
    shortName: "ASR 1.7B",
    fullName: "Qwen3-ASR 1.7B",
    description: "High-quality speech-to-text model supporting 52 languages",
    features: [
      "52 languages",
      "Language detection",
      "State-of-the-art accuracy",
    ],
    size: "4.4 GB",
  },
  "Qwen3-ASR-1.7B-4bit": {
    shortName: "ASR 1.7B 4-bit",
    fullName: "Qwen3-ASR 1.7B (MLX 4-bit)",
    description: "Quantized 1.7B ASR for RTX 4090 / M3 workloads",
    features: ["52 languages", "4-bit quantized", "Large-context"],
    size: "1.5 GB",
    quantization: "4-bit",
  },
  "Qwen3-ASR-1.7B-8bit": {
    shortName: "ASR 1.7B 8-bit",
    fullName: "Qwen3-ASR 1.7B (MLX 8-bit)",
    description: "8-bit ASR for high fidelity transcripts on Apple Silicon",
    features: ["52 languages", "8-bit quantized", "Large-context"],
    size: "2.3 GB",
    quantization: "8-bit",
  },
  "Qwen3-ASR-1.7B-bf16": {
    shortName: "ASR 1.7B BF16",
    fullName: "Qwen3-ASR 1.7B (MLX bf16)",
    description: "BF16 ASR providing maximum quality and accuracy",
    features: ["52 languages", "bf16 weights", "Large-context"],
    size: "3.8 GB",
    quantization: "bf16",
  },
  "Qwen3-ForcedAligner-0.6B": {
    shortName: "ForcedAligner 0.6B",
    fullName: "Qwen3-ForcedAligner 0.6B",
    description: "Aligns transcript text to precise speech timestamps",
    features: ["Forced alignment", "Word-level timestamps", "Qwen3 pipeline"],
    size: "1.7 GB",
  },
  "Qwen3-ForcedAligner-0.6B-4bit": {
    shortName: "ForcedAligner 0.6B 4-bit",
    fullName: "Qwen3-ForcedAligner 0.6B (MLX 4-bit)",
    description: "Quantized forced aligner for low-memory timestamp alignment",
    features: ["Forced alignment", "Word-level timestamps", "4-bit quantized"],
    size: "0.7 GB",
    quantization: "4-bit",
  },
  "Parakeet-TDT-0.6B-v2": {
    shortName: "Parakeet v2",
    fullName: "Parakeet-TDT 0.6B v2",
    description: "English FastConformer-TDT ASR model distributed as .nemo",
    features: ["English ASR", "Word timestamps", ".nemo checkpoint"],
    size: "4.6 GB",
  },
  "Parakeet-TDT-0.6B-v3": {
    shortName: "Parakeet v3",
    fullName: "Parakeet-TDT 0.6B v3",
    description:
      "Multilingual FastConformer-TDT ASR model distributed as .nemo",
    features: ["Multilingual ASR", "Word timestamps", ".nemo checkpoint"],
    size: "9.3 GB",
  },
  "diar_streaming_sortformer_4spk-v2.1": {
    shortName: "Sortformer 4spk",
    fullName: "Streaming Sortformer 4spk v2.1",
    description:
      "Streaming speaker diarization model from NVIDIA in .nemo format",
    features: ["Diarization", "Up to 4 speakers", "Streaming"],
    size: "0.5 GB",
  },
  "Voxtral-Mini-4B-Realtime-2602": {
    shortName: "Voxtral 4B",
    fullName: "Voxtral Mini 4B Realtime",
    description:
      "Realtime streaming ASR model from Mistral AI with high-quality transcription",
    features: [
      "Realtime streaming",
      "High-quality transcription",
      "Multilingual support",
      "Causal attention for streaming",
    ],
    size: "~8 GB",
  },
};

export function ModelManager({
  models,
  selectedModel,
  onDownload,
  onLoad,
  onUnload,
  onDelete,
  onSelect,
  onCancelDownload,
  downloadProgress,
  modelFilter,
  isModelDisabled,
  disabledModelLabel,
  emptyStateTitle,
  emptyStateDescription,
}: ModelManagerProps) {
  const [expandedModel, setExpandedModel] = useState<string | null>(null);
  const [pendingDeleteVariant, setPendingDeleteVariant] = useState<
    string | null
  >(null);
  const ttsModels = models
    .filter((m) => !m.variant.includes("Tokenizer"))
    .filter((m) => (modelFilter ? modelFilter(m.variant) : true))
    .sort((a, b) => {
      // Sort by size (smallest to largest)
      const sizeA = getModelSizeBytes(a);
      const sizeB = getModelSizeBytes(b);
      if (sizeA !== sizeB) {
        return sizeA - sizeB;
      }
      // If sizes are equal, sort by name
      return a.variant.localeCompare(b.variant);
    });

  if (ttsModels.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-center bg-card rounded-lg border shadow-sm">
        <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-3">
          <Download className="w-5 h-5 text-muted-foreground" />
        </div>
        <h3 className="text-sm font-semibold text-foreground mb-1">
          {emptyStateTitle || "No Models Available"}
        </h3>
        <p className="text-xs text-muted-foreground max-w-[200px]">
          {emptyStateDescription || "Download models to get started"}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {ttsModels.map((model) => {
        const details = MODEL_DETAILS[model.variant] || {
          shortName: model.variant,
          fullName: model.variant,
          description: "",
          features: [],
          size: "",
        };
        const displayName = withQwen3Prefix(details.shortName, model.variant);

        const isSelected = selectedModel === model.variant;
        const isExpanded = expandedModel === model.variant;
        const isDownloading = model.status === "downloading";
        const isLoading = model.status === "loading";
        const isReady = model.status === "ready";
        const isDownloaded = model.status === "downloaded";
        const isDisabled = isModelDisabled?.(model.variant) ?? false;
        const progressValue = downloadProgress[model.variant];
        const progress = progressValue?.percent ?? model.download_progress ?? 0;

        return (
          <div
            key={model.variant}
            className={cn(
              "border rounded-lg transition-colors shadow-sm",
              isDisabled && "opacity-50",
              isSelected
                ? "border-primary/50 bg-primary/5 shadow-md"
                : "border-border bg-card",
            )}
          >
            {/* Main card */}
            <div
              className={cn(
                "p-3 rounded-lg",
                isDisabled ? "cursor-not-allowed" : "cursor-pointer",
                !isDisabled && !isExpanded && "hover:bg-muted/50",
              )}
              aria-disabled={isDisabled}
              onClick={() => {
                if (isDisabled) {
                  return;
                }
                if (isReady && !isSelected) {
                  onSelect(model.variant);
                }
                setExpandedModel(isExpanded ? null : model.variant);
              }}
            >
              <div className="flex items-center gap-3">
                {/* Status indicator */}
                <div className="flex-shrink-0">
                  {isDownloading ? (
                    <div className="relative w-8 h-8">
                      <svg className="w-8 h-8 transform -rotate-90">
                        <circle
                          cx="16"
                          cy="16"
                          r="14"
                          fill="none"
                          stroke="currentColor"
                          className="text-muted"
                          strokeWidth="2"
                        />
                        <circle
                          cx="16"
                          cy="16"
                          r="14"
                          fill="none"
                          stroke="currentColor"
                          className="text-primary"
                          strokeWidth="2"
                          strokeDasharray={`${2 * Math.PI * 14}`}
                          strokeDashoffset={`${2 * Math.PI * 14 * (1 - progress / 100)}`}
                          strokeLinecap="round"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center text-[10px] text-foreground font-medium">
                        {Math.round(progress)}
                      </div>
                    </div>
                  ) : isLoading ? (
                    <Loader2 className="w-5 h-5 text-primary animate-spin" />
                  ) : (
                    <div
                      className={cn(
                        "w-2.5 h-2.5 rounded-full shadow-sm",
                        isReady && "bg-green-500",
                        isDownloaded && "bg-muted-foreground",
                        model.status === "not_downloaded" && "bg-muted",
                      )}
                    />
                  )}
                </div>

                {/* Model info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-foreground tracking-tight">
                      {displayName}
                    </span>
                    {isDisabled && (
                      <span className="text-[10px] px-1.5 py-0.5 bg-amber-500/10 text-amber-500 rounded font-medium border border-amber-500/20">
                        {disabledModelLabel || "DISABLED"}
                      </span>
                    )}
                    {isSelected && (
                      <span className="text-[10px] px-1.5 py-0.5 bg-primary/10 text-primary rounded font-semibold border border-primary/20 tracking-wider">
                        ACTIVE
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-muted-foreground mt-0.5 font-medium">
                    {getModelSizeLabel(model)}
                    {isDownloading && progressValue && (
                      <>
                        {" "}
                        • {progress.toFixed(0)}% (
                        {formatBytes(progressValue.downloadedBytes)} /{" "}
                        {formatBytes(progressValue.totalBytes)})
                      </>
                    )}
                    {isLoading && " • Loading..."}
                  </div>
                </div>

                {/* Expand icon */}
                <ChevronRight
                  className={cn(
                    "w-4 h-4 text-muted-foreground transition-transform flex-shrink-0",
                    isExpanded && "rotate-90",
                  )}
                />
              </div>

              {/* Progress bar */}
              {isDownloading && (
                <div className="mt-2 h-1.5 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              )}
            </div>

            {/* Expanded details */}
            <AnimatePresence>
              {isExpanded && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden border-t bg-muted/20 rounded-b-lg"
                >
                  <div className="p-4 space-y-4">
                    {/* Full name */}
                    <div>
                      <div className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                        Model
                      </div>
                      <div className="text-sm font-semibold tracking-tight">
                        {details.fullName}
                      </div>
                    </div>

                    {/* Description */}
                    <div>
                      <div className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                        Description
                      </div>
                      <div className="text-sm text-foreground/80 leading-relaxed">
                        {details.description}
                      </div>
                    </div>

                    {/* Features */}
                    {details.features.length > 0 && (
                      <div>
                        <div className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider mb-1">
                          Features
                        </div>
                        <div className="flex flex-wrap gap-1.5 mt-2">
                          {details.features.map((feature, i) => (
                            <span
                              key={i}
                              className="text-[11px] px-2 py-0.5 bg-background border text-muted-foreground font-medium rounded-md shadow-sm"
                            >
                              {feature}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Actions */}
                    <div className="flex flex-wrap items-center gap-2 pt-3 border-t">
                      {isDisabled && (
                        <div className="text-xs font-medium text-amber-500 w-full mb-2">
                          This model is unavailable in this view.
                        </div>
                      )}

                      {isDownloading && onCancelDownload && (
                        <Button
                          onClick={(e) => {
                            e.stopPropagation();
                            onCancelDownload(model.variant);
                          }}
                          variant="destructive"
                          size="sm"
                          className="flex-1 gap-1.5 h-8 shadow-sm"
                          disabled={isDisabled}
                        >
                          <X className="w-3.5 h-3.5" />
                          Cancel Download
                        </Button>
                      )}

                      {model.status === "not_downloaded" &&
                        (requiresManualDownload(model.variant) ? (
                          <Button
                            variant="secondary"
                            size="sm"
                            className="flex-1 gap-1.5 h-8 shadow-sm"
                            disabled
                            title="Manual download required. See docs/user/manual-gemma-3-1b-download.md."
                          >
                            <Download className="w-3.5 h-3.5" />
                            Manual download
                          </Button>
                        ) : (
                          <Button
                            onClick={(e) => {
                              e.stopPropagation();
                              onDownload(model.variant);
                            }}
                            size="sm"
                            className="flex-1 gap-1.5 h-8 shadow-sm"
                            disabled={isDisabled}
                          >
                            <Download className="w-3.5 h-3.5" />
                            Download
                          </Button>
                        ))}

                      {isDownloaded && (
                        <>
                          <Button
                            onClick={(e) => {
                              e.stopPropagation();
                              onLoad(model.variant);
                            }}
                            size="sm"
                            className="flex-1 gap-1.5 h-8 shadow-sm"
                            disabled={isDisabled}
                          >
                            <Play className="w-3.5 h-3.5" />
                            Load
                          </Button>
                          {pendingDeleteVariant === model.variant ? (
                            <div className="flex items-center gap-1">
                              <Button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setPendingDeleteVariant(null);
                                  onDelete(model.variant);
                                }}
                                variant="destructive"
                                size="sm"
                                className="h-8 px-2.5 shadow-sm"
                                disabled={isDisabled}
                              >
                                <Check className="w-4 h-4" />
                              </Button>
                              <Button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setPendingDeleteVariant(null);
                                }}
                                variant="outline"
                                size="sm"
                                className="h-8 px-2.5 shadow-sm"
                                disabled={isDisabled}
                              >
                                <X className="w-4 h-4" />
                              </Button>
                            </div>
                          ) : (
                            <Button
                              onClick={(e) => {
                                e.stopPropagation();
                                setPendingDeleteVariant(model.variant);
                              }}
                              variant="destructive"
                              size="sm"
                              className="h-8 px-2.5 shadow-sm opacity-80 hover:opacity-100"
                              disabled={isDisabled}
                            >
                              <Trash2 className="w-4 h-4" />
                            </Button>
                          )}
                        </>
                      )}

                      {isReady && (
                        <>
                          <Button
                            onClick={(e) => {
                              e.stopPropagation();
                              onUnload(model.variant);
                            }}
                            variant="secondary"
                            size="sm"
                            className="flex-1 gap-1.5 h-8 shadow-sm"
                            disabled={isDisabled}
                          >
                            <Square className="w-3.5 h-3.5" />
                            Unload
                          </Button>
                          {pendingDeleteVariant === model.variant ? (
                            <div className="flex items-center gap-1">
                              <Button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setPendingDeleteVariant(null);
                                  onDelete(model.variant);
                                }}
                                variant="destructive"
                                size="sm"
                                className="h-8 px-2.5 shadow-sm"
                                disabled={isDisabled}
                              >
                                <Check className="w-4 h-4" />
                              </Button>
                              <Button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setPendingDeleteVariant(null);
                                }}
                                variant="outline"
                                size="sm"
                                className="h-8 px-2.5 shadow-sm"
                                disabled={isDisabled}
                              >
                                <X className="w-4 h-4" />
                              </Button>
                            </div>
                          ) : (
                            <Button
                              onClick={(e) => {
                                e.stopPropagation();
                                setPendingDeleteVariant(model.variant);
                              }}
                              variant="destructive"
                              size="sm"
                              className="h-8 px-2.5 shadow-sm opacity-80 hover:opacity-100"
                              disabled={isDisabled}
                            >
                              <Trash2 className="w-4 h-4" />
                            </Button>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        );
      })}
    </div>
  );
}
