import { Fragment } from "react";
import {
  Check,
  Loader2,
  MoreHorizontal,
  RotateCcw,
  Square,
  Trash2,
  Volume2,
} from "lucide-react";

import type { ChatGenerationStats, ChatThreadMessageRecord } from "@/api";
import {
  MessageTime,
  ReplyNotice,
  StoppedChip,
} from "@/features/chat/components/messages/MessageParts";
import { ReasoningBlock } from "@/features/chat/components/messages/ReasoningBlock";
import { cn } from "@/lib/utils";
import { CopyButton } from "@/components/ui/CopyButton";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { MarkdownContent } from "@/components/ui/MarkdownContent";
import {
  parseAssistantContent,
  stripThinkingArtifacts,
  summarizeChatStats,
} from "@/features/chat/playground/support";

const footerButtonClass =
  "inline-flex h-7 w-7 items-center justify-center rounded-lg text-[var(--text-subtle)] transition-colors hover:bg-[var(--surface-raised)] hover:text-[var(--text-primary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--border-strong)]";

function speedTitle(stats: ReturnType<typeof summarizeChatStats>): string {
  const parts = [
    stats.decodeOnly
      ? "Decode speed: generated tokens over decode time, excluding queue wait and prompt processing"
      : "Generated tokens over total time, including queue wait and prompt processing",
    `Total ${Math.round(stats.totalMs)} ms`,
  ];
  if (stats.ttftMs !== null) {
    parts.push(`First token after ${Math.round(stats.ttftMs)} ms`);
  }
  return parts.join(". ");
}

interface AssistantMessageProps {
  message: ChatThreadMessageRecord;
  /** This reply is the one streaming now. */
  isStreamingReply: boolean;
  /** The newest reply in the thread keeps its footer visible. */
  isLatestReply: boolean;
  /** Display name of the model that wrote the reply. */
  modelLabel: string | null;
  /** Engine stats from streaming this reply in this session. */
  stats: ChatGenerationStats | null;
  readAloudState: "idle" | "loading" | "playing";
  onReadAloud: (text: string) => void;
  /** Offer Retry: the thread's last message, while no turn is running. */
  canRetry: boolean;
  /** Loaded chat models for Retry with…. */
  retryModels: { value: string; label: string }[];
  currentModelId: string | null;
  onRetry: (modelId?: string) => void;
  /** Absent while a turn is running. */
  onDelete?: () => void;
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
  modelLabel,
  stats,
  readAloudState,
  onReadAloud,
  canRetry,
  retryModels,
  currentModelId,
  onRetry,
  onDelete,
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
  // While the answer streams, the caret trails its last line (see
  // .chat-streaming-answer); before any text it stands on its own.
  const showInlineCaret =
    isStreamingReply && !reasoningActive && answerText.length > 0;
  const turnStats =
    stats ??
    (typeof message.tokens_generated === "number" &&
    typeof message.generation_time_ms === "number"
      ? {
          tokens_generated: message.tokens_generated,
          generation_time_ms: message.generation_time_ms,
        }
      : null);
  const statsSummary = turnStats ? summarizeChatStats(turnStats) : null;
  const details = [
    modelLabel ? (
      <span key="model" className="min-w-0 truncate">
        {modelLabel}
      </span>
    ) : null,
    statsSummary ? (
      <span key="tokens" className="whitespace-nowrap tabular-nums">
        {statsSummary.tokens} tokens
      </span>
    ) : null,
    statsSummary && statsSummary.tokensPerSecond > 0 ? (
      <span
        key="speed"
        className="whitespace-nowrap tabular-nums"
        title={speedTitle(statsSummary)}
      >
        {Math.round(statsSummary.tokensPerSecond)} tok/s
      </span>
    ) : null,
  ].filter((detail) => detail !== null);
  const finishReason = isStreamingReply ? null : message.finish_reason;

  return (
    <article
      aria-label="Reply"
      className={cn(
        "group flex flex-col gap-2.5 break-words text-[0.9375rem] leading-[1.7] text-[var(--text-primary)]",
        showInlineCaret && "chat-streaming-answer",
      )}
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
        isStreamingReply || finishReason === "cancelled" ? null : (
          <div className="italic text-muted-foreground opacity-70">
            No final answer was generated.
          </div>
        )
      ) : displayContent.trim().length > 0 ? (
        <MarkdownContent content={displayContent} />
      ) : !isStreamingReply && finishReason !== "cancelled" ? (
        <div className="italic text-muted-foreground opacity-70">
          No visible answer was generated. Please retry the message.
        </div>
      ) : null}

      {finishReason === "cancelled" ? <StoppedChip /> : null}
      {finishReason === "length" ? (
        <ReplyNotice tone="warning">
          This reply reached the length limit, so it may be cut off.
        </ReplyNotice>
      ) : null}
      {finishReason === "error" ? (
        <ReplyNotice tone="danger">
          The model hit an error before finishing this reply.
        </ReplyNotice>
      ) : null}

      {isStreamingReply && !reasoningActive && !showInlineCaret ? (
        <span className="chat-caret" aria-hidden="true" />
      ) : null}
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
          {answerText ? (
            <button
              type="button"
              onClick={() => onReadAloud(answerText)}
              aria-label={
                readAloudState === "idle"
                  ? "Read aloud"
                  : readAloudState === "loading"
                    ? "Preparing audio, select to cancel"
                    : "Stop reading"
              }
              title={readAloudState === "idle" ? "Read aloud" : "Stop"}
              className={footerButtonClass}
            >
              {readAloudState === "loading" ? (
                <Loader2
                  className="h-[15px] w-[15px] animate-spin"
                  aria-hidden="true"
                />
              ) : readAloudState === "playing" ? (
                <Square className="h-3 w-3 fill-current" aria-hidden="true" />
              ) : (
                <Volume2 className="h-[15px] w-[15px]" aria-hidden="true" />
              )}
            </button>
          ) : null}
          {canRetry ? (
            <button
              type="button"
              onClick={() => onRetry()}
              aria-label="Retry"
              title="Retry"
              className={footerButtonClass}
            >
              <RotateCcw className="h-[15px] w-[15px]" aria-hidden="true" />
            </button>
          ) : null}
          {(canRetry && retryModels.length > 0) || onDelete ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  type="button"
                  aria-label="More reply actions"
                  title="More"
                  className={footerButtonClass}
                >
                  <MoreHorizontal
                    className="h-[15px] w-[15px]"
                    aria-hidden="true"
                  />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-48">
                {canRetry && retryModels.length > 0 ? (
                  <DropdownMenuSub>
                    <DropdownMenuSubTrigger className="gap-2">
                      <RotateCcw className="h-4 w-4" aria-hidden="true" />
                      Retry with…
                    </DropdownMenuSubTrigger>
                    <DropdownMenuSubContent className="w-56">
                      {retryModels.map((model) => (
                        <DropdownMenuItem
                          key={model.value}
                          onSelect={() => onRetry(model.value)}
                          className="justify-between gap-2"
                        >
                          <span className="truncate">{model.label}</span>
                          {model.value === currentModelId ? (
                            <Check
                              className="h-4 w-4 shrink-0"
                              aria-label="Current model"
                            />
                          ) : null}
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuSubContent>
                  </DropdownMenuSub>
                ) : null}
                {canRetry && retryModels.length > 0 && onDelete ? (
                  <DropdownMenuSeparator />
                ) : null}
                {onDelete ? (
                  <DropdownMenuItem
                    onSelect={onDelete}
                    className="gap-2 text-[var(--danger-text)] focus:text-[var(--danger-text)]"
                  >
                    <Trash2 className="h-4 w-4" aria-hidden="true" />
                    Delete reply
                  </DropdownMenuItem>
                ) : null}
              </DropdownMenuContent>
            </DropdownMenu>
          ) : null}
          {answerText ? (
            <span
              aria-hidden="true"
              className="mx-1.5 h-3.5 w-px shrink-0 bg-[var(--border-subtle)]"
            />
          ) : null}
          <span className="flex min-w-0 items-center gap-1.5 px-1">
            {details.map((detail, index) => (
              <Fragment key={detail.key}>
                {index > 0 ? <span aria-hidden="true">·</span> : null}
                {detail}
              </Fragment>
            ))}
            {details.length > 0 ? <span aria-hidden="true">·</span> : null}
          </span>
          <MessageTime timestamp={message.created_at} />
        </footer>
      )}
    </article>
  );
}
