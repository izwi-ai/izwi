import type { ChatThreadMessageRecord } from "@/api";
import { MessageTime } from "@/features/chat/components/messages/MessageParts";
import { ReasoningBlock } from "@/features/chat/components/messages/ReasoningBlock";
import { cn } from "@/lib/utils";
import { CopyButton } from "@/components/ui/CopyButton";
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
  /** When the streaming turn started; null for finished replies. */
  turnStartedAt: number | null;
  /** Reasoning time measured in this session, if any. */
  reasoningMs: number | null;
  onReasoningFinished: (durationMs: number) => void;
}

export function AssistantMessage({
  message,
  isStreamingReply,
  isLatestReply,
  implicitThink,
  thinkingEnabled,
  isThoughtExpanded,
  onToggleThought,
  turnStartedAt,
  reasoningMs,
  onReasoningFinished,
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
  const answerStarted = !!parsed && parsed.answer.length > 0;
  // Reasoning is live until the answer starts. Before the first token of a
  // thinking turn the row already shows, so the wait reads as thinking.
  const reasoningActive =
    !!parsed &&
    isStreamingReply &&
    (parsed.hasIncompleteThink || !answerStarted) &&
    (parsed.thinking.length > 0 || thinkingEnabled);
  const showReasoning =
    reasoningActive ||
    (!!parsed && parsed.hasThink && parsed.thinking.length > 0);
  const answerText = (parsed ? parsed.answer : displayContent).trim();

  return (
    <article
      aria-label="Reply"
      className="group flex flex-col gap-2.5 break-words text-[0.9375rem] leading-[1.7] text-[var(--text-primary)]"
    >
      {showReasoning && parsed && (
        <ReasoningBlock
          thinking={parsed.thinking}
          active={reasoningActive}
          startedAt={turnStartedAt}
          durationMs={reasoningMs}
          expanded={isThoughtExpanded}
          onToggle={onToggleThought}
          onFinished={onReasoningFinished}
        />
      )}

      {answerStarted && parsed ? (
        <MarkdownContent content={parsed.answer} />
      ) : parsed && parsed.hasThink ? (
        isStreamingReply ? null : (
          <div className="italic text-muted-foreground opacity-70">
            No final answer was generated.
          </div>
        )
      ) : displayContent.trim().length > 0 ? (
        <MarkdownContent content={displayContent} />
      ) : !isStreamingReply ? (
        <div className="italic text-muted-foreground opacity-70">
          No visible answer was generated. Please retry the message.
        </div>
      ) : null}

      {isStreamingReply && !reasoningActive && (
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
          {answerText ? (
            <CopyButton text={answerText} label="Copy reply" />
          ) : null}
          <MessageTime timestamp={message.created_at} />
        </footer>
      )}
    </article>
  );
}
