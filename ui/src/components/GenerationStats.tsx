import { motion } from "framer-motion";
import { Clock, Zap, Timer, Hash } from "lucide-react";
import clsx from "clsx";

export interface TTSStats {
  generation_time_ms: number;
  audio_duration_secs: number;
  rtf: number;
  tokens_generated: number;
}

export interface ASRStats {
  processing_time_ms: number;
  audio_duration_secs: number | null;
  rtf: number | null;
}

interface GenerationStatsProps {
  stats: TTSStats | ASRStats | null;
  type: "tts" | "asr";
  className?: string;
  surface?: "card" | "plain";
}

function isTTSStats(stats: TTSStats | ASRStats): stats is TTSStats {
  return "tokens_generated" in stats;
}

export function GenerationStats({
  stats,
  type,
  className,
  surface = "card",
}: GenerationStatsProps) {
  if (!stats) return null;

  const processingTime = isTTSStats(stats)
    ? stats.generation_time_ms
    : stats.processing_time_ms;

  const audioDuration = isTTSStats(stats)
    ? stats.audio_duration_secs
    : stats.audio_duration_secs;

  const rtf = isTTSStats(stats) ? stats.rtf : stats.rtf;

  // RTF < 1 means faster than real-time (good!)
  const rtfStatus =
    rtf !== null
      ? rtf < 0.5
        ? "excellent"
        : rtf < 1.0
          ? "good"
          : rtf < 2.0
            ? "fair"
            : "slow"
      : null;

  const rtfColors = {
    excellent: "text-gray-300 bg-white/10",
    good: "text-gray-300 bg-white/10",
    fair: "text-yellow-400 bg-yellow-500/10",
    slow: "text-red-400 bg-red-500/10",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 5 }}
      animate={{ opacity: 1, y: 0 }}
      className={clsx(
        "flex flex-wrap items-center gap-3 px-3 py-2",
        surface === "card"
          ? "rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-2)]"
          : null,
        className,
      )}
    >
      {/* Processing Time */}
      <div className="flex items-center gap-1.5">
        <Clock className="w-3.5 h-3.5 text-gray-500" />
        <span className="text-xs text-gray-400">
          {processingTime < 1000
            ? `${processingTime.toFixed(0)}ms`
            : `${(processingTime / 1000).toFixed(2)}s`}
        </span>
      </div>

      {/* Audio Duration */}
      {audioDuration !== null && (
        <div className="flex items-center gap-1.5">
          <Timer className="w-3.5 h-3.5 text-gray-500" />
          <span className="text-xs text-gray-400">
            {audioDuration.toFixed(2)}s audio
          </span>
        </div>
      )}

      {/* RTF (Real-Time Factor) */}
      {rtf !== null && rtfStatus && (
        <div
          className={clsx(
            "flex items-center gap-1.5 px-2 py-0.5 rounded",
            rtfColors[rtfStatus]
          )}
        >
          <Zap className="w-3.5 h-3.5" />
          <span className="text-xs font-medium">
            {rtf < 1 ? `${rtf.toFixed(2)}x` : `${rtf.toFixed(2)}x`}
          </span>
          <span className="text-[10px] opacity-75">
            {rtf < 1 ? "faster" : "RTF"}
          </span>
        </div>
      )}

      {/* Tokens (TTS only) */}
      {type === "tts" && isTTSStats(stats) && stats.tokens_generated > 0 && (
        <div className="flex items-center gap-1.5">
          <Hash className="w-3.5 h-3.5 text-gray-500" />
          <span className="text-xs text-gray-400">
            {stats.tokens_generated} tokens
          </span>
        </div>
      )}
    </motion.div>
  );
}
