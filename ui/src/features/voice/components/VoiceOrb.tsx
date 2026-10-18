import { motion, useReducedMotion } from "framer-motion";
import { Mic } from "lucide-react";

import { cn } from "@/lib/utils";
import type { RuntimeStatus } from "@/features/voice/realtime/support";

const BAR_HEIGHTS = [22, 44, 60, 36, 18];

/**
 * The conversation's one visual: quiet when idle, a halo that follows your
 * voice while you speak, dots while thinking and bars while Izwi speaks.
 * Decorative; the status line next to it carries the meaning.
 */
export function VoiceOrb({
  status,
  audioLevel,
}: {
  status: RuntimeStatus;
  audioLevel: number;
}) {
  const reduceMotion = useReducedMotion();
  const level = Math.min(1, audioLevel * 2.5);
  const speaking = status === "assistant_speaking";
  const userSpeaking = status === "user_speaking";
  const idle = status === "idle";

  return (
    <div
      aria-hidden="true"
      className="relative flex h-[264px] w-[264px] items-center justify-center"
    >
      {!idle && (
        <div
          className={cn(
            "absolute rounded-full transition-[width,height,background-color,border-color] duration-200",
            userSpeaking || speaking
              ? "bg-[var(--accent-soft)]"
              : "border border-[var(--border-muted)]",
            status === "preparing" && !reduceMotion && "animate-pulse",
          )}
          style={{
            width: userSpeaking ? 212 + level * 52 : speaking ? 236 : 212,
            height: userSpeaking ? 212 + level * 52 : speaking ? 236 : 212,
          }}
        />
      )}
      <div
        className={cn(
          "relative flex h-[184px] w-[184px] items-center justify-center rounded-full transition-colors duration-300",
          idle &&
            "border border-[var(--border-default)] bg-[var(--bg-surface-2)] text-[var(--text-muted)] shadow-[inset_0_0_0_12px_var(--bg-surface-0),inset_0_0_0_13px_var(--border-muted)]",
          (status === "listening" || status === "preparing") &&
            "border border-[var(--border-strong)] bg-[var(--bg-surface-3)] text-[var(--text-primary)]",
          userSpeaking &&
            "border border-[var(--border-strong)] bg-[var(--bg-surface-4)] text-[var(--text-primary)] shadow-[0_0_0_14px_var(--accent-strong)]",
          status === "processing" &&
            "border border-[var(--border-strong)] bg-[var(--bg-surface-3)]",
          speaking && "bg-[var(--accent-solid)]",
        )}
      >
        {status === "processing" ? (
          <div className="flex gap-2.5">
            {[0, 1, 2].map((index) => (
              <motion.span
                key={index}
                className="h-2.5 w-2.5 rounded-full bg-[var(--text-primary)]"
                animate={reduceMotion ? undefined : { opacity: [0.3, 1, 0.3] }}
                transition={{
                  duration: 1.2,
                  repeat: Infinity,
                  delay: index * 0.2,
                }}
              />
            ))}
          </div>
        ) : speaking ? (
          <div className="flex items-center gap-[7px]">
            {BAR_HEIGHTS.map((height, index) => (
              <motion.span
                key={index}
                className="w-2 rounded-full bg-[var(--text-on-accent)]"
                style={{ height }}
                animate={
                  reduceMotion
                    ? undefined
                    : { scaleY: [0.45, 1, 0.45] }
                }
                transition={{
                  duration: 1,
                  repeat: Infinity,
                  delay: index * 0.1,
                  ease: "easeInOut",
                }}
              />
            ))}
          </div>
        ) : (
          <Mic className="h-8 w-8" strokeWidth={1.75} />
        )}
      </div>
    </div>
  );
}
