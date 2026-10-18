import type { Ref, UIEventHandler } from "react";
import { ChevronDown, ChevronRight, Loader2 } from "lucide-react";

import type { ChatThreadMessageRecord } from "@/api";
import { MessageTime } from "@/features/chat/components/messages/MessageParts";
import { cn } from "@/lib/utils";
import { MarkdownContent } from "@/components/ui/MarkdownContent";
import {
  parseAssistantContent,
  stripThinkingArtifacts,
} from "@/features/chat/playground/support";

interface AssistantMessageProps {
  message: ChatThreadMessageRecord;
  /** This reply is the one streaming now. */
  isStreamingReply: boolean;
  /** The newest reply in the thread keeps its footer visible. */
  isLatestReply: boolean;
  /** The model that wrote this reply emits reasoning before `</think>`. */
  implicitThink: boolean;
  /** Thinking is on for the turn being streamed. */
  thinkingEnabled: boolean;
  isThoughtExpanded: boolean;
  onToggleThought: () => void;
  streamingThinkingRef: Ref<HTMLDivElement>;
  onStreamingThinkingScroll: UIEventHandler<HTMLDivElement>;
}

export function AssistantMessage({
  message,
  isStreamingReply,
  isLatestReply,
  implicitThink,
  thinkingEnabled,
  isThoughtExpanded,
  onToggleThought,
  streamingThinkingRef,
  onStreamingThinkingScroll,
}: AssistantMessageProps) {
  // Models without thinking support never show reasoning: stray think tags
  // from them are artifacts to hide.
  const showsReasoning = implicitThink || (isStreamingReply && thinkingEnabled);
  const displayContent = showsReasoning
    ? message.content
    : stripThinkingArtifacts(message.content || "");
  const parsed = showsReasoning
    ? parseAssistantContent(message.content || "", {
        implicitOpenThinkTag: implicitThink,
        treatNoTagAsThinking: isStreamingReply && thinkingEnabled,
      })
    : null;
  const showStreamingThinking =
    !!parsed &&
    isStreamingReply &&
    parsed.thinking.length > 0 &&
    (parsed.hasIncompleteThink || parsed.answer.length === 0);
  const showAnswerOnly =
    !!parsed &&
    parsed.answer.length > 0 &&
    parsed.hasThink &&
    !showStreamingThinking;

  return (
    <article
      aria-label="Reply"
      className="group flex flex-col gap-2.5 break-words text-[0.9375rem] leading-[1.7] text-[var(--text-primary)]"
    >
      {showStreamingThinking && parsed && (
        <div className="mb-3 rounded-lg bg-muted/50 border border-[var(--border-muted)] px-3 py-2 text-xs text-muted-foreground">
          <div className="mb-2 flex items-center gap-1.5 uppercase tracking-wider text-[10px] font-semibold">
            <Loader2 className="w-3 h-3 animate-spin text-primary" />
            Thinking
          </div>
          <div
            ref={streamingThinkingRef}
            onScroll={onStreamingThinkingScroll}
            className="h-40 overflow-y-auto whitespace-pre-wrap rounded-md border border-[color-mix(in_srgb,var(--border-muted)_70%,transparent)] bg-background/60 px-2.5 py-2 font-mono text-[11px] leading-relaxed scrollbar-thin sm:h-48"
          >
            {parsed.thinking}
          </div>
        </div>
      )}

      {parsed && parsed.answer.length > 0 ? (
        <MarkdownContent content={parsed.answer} />
      ) : parsed && parsed.hasThink ? (
        <div className="italic text-muted-foreground opacity-70">
          {isStreamingReply ? "Thinking..." : "No final answer was generated."}
        </div>
      ) : displayContent.trim().length > 0 ? (
        <MarkdownContent content={displayContent} />
      ) : !isStreamingReply ? (
        <div className="italic text-muted-foreground opacity-70">
          No visible answer was generated. Please retry the message.
        </div>
      ) : null}

      {parsed && parsed.hasThink && !showStreamingThinking && (
        <div className="mt-3 border-t pt-2">
          <button
            onClick={onToggleThought}
            className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground font-medium transition-colors"
          >
            {isThoughtExpanded ? (
              <ChevronDown className="w-3 h-3" />
            ) : (
              <ChevronRight className="w-3 h-3" />
            )}
            {isThoughtExpanded
              ? "Hide thought process"
              : "Show thought process"}
          </button>
        </div>
      )}

      {parsed &&
        parsed.hasThink &&
        !showStreamingThinking &&
        isThoughtExpanded && (
          <div className="mt-2 h-40 overflow-y-auto whitespace-pre-wrap rounded-lg border border-[var(--border-muted)] bg-muted/30 px-3 py-2 text-xs font-mono text-[11px] leading-relaxed text-muted-foreground scrollbar-thin sm:h-48">
            {parsed.thinking}
          </div>
        )}

      {isStreamingReply &&
        ((parsed && parsed.answer.length > 0) || !showAnswerOnly) && (
          <span className="inline-flex items-center ml-2 align-middle">
            <span className="w-1.5 h-4 bg-primary animate-pulse inline-block" />
          </span>
        )}
      {!isStreamingReply && (
        <footer
          className={cn(
            "-ml-1.5 flex h-7 items-center gap-0.5 text-[11.5px] text-[var(--text-subtle)] transition-opacity",
            !isLatestReply &&
              "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100",
          )}
        >
          <MessageTime timestamp={message.created_at} />
        </footer>
      )}
    </article>
  );
}
