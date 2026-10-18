import { Film, ImageIcon } from "lucide-react";

import type { ChatThreadMessageRecord } from "@/api";
import { CopyButton } from "@/components/ui/CopyButton";
import { MarkdownContent } from "@/components/ui/MarkdownContent";
import {
  type ImagePreviewState,
  parseUserMessageDisplayFromContentParts,
} from "@/features/chat/playground/support";
import { cn } from "@/lib/utils";
import { MessageTime } from "@/features/chat/components/messages/MessageParts";

interface UserMessageProps {
  message: ChatThreadMessageRecord;
  onOpenImage: (preview: ImagePreviewState) => void;
}

/** A sent message in its bubble, with its time shown on hover or focus. */
export function UserMessage({ message, onOpenImage }: UserMessageProps) {
  const parsed = parseUserMessageDisplayFromContentParts(message);
  const text =
    parsed && parsed.attachments.length > 0 ? parsed.text : message.content;

  return (
    <article
      aria-label="Your message"
      className="group flex max-w-[78%] flex-col items-end gap-1 self-end"
    >
      <div className="break-words rounded-[18px] bg-[var(--surface-raised)] px-4 py-2.5 text-[0.9375rem] leading-relaxed text-[var(--text-primary)]">
        {(!parsed || parsed.text || parsed.attachments.length === 0) && (
          <MarkdownContent
            content={text}
            className="leading-relaxed [&_p]:m-0 [&_p]:bg-transparent [&_mark]:bg-transparent [&_code]:bg-transparent [&_code]:border-0 [&_code]:px-0 [&_code]:py-0 [&_code]:font-inherit"
          />
        )}
        {parsed && parsed.attachments.length > 0 && (
          <div className={cn("flex flex-wrap gap-2", parsed.text && "mt-2")}>
            {parsed.attachments.map((attachment, attachmentIndex) => (
              <div key={`${message.id}-attachment-${attachmentIndex}`}>
                {attachment.kind === "image" && attachment.source ? (
                  <button
                    type="button"
                    onClick={() =>
                      onOpenImage({
                        source: attachment.source!,
                        label: attachment.label,
                      })
                    }
                    title={`Open ${attachment.label}`}
                    className="block h-16 w-16 overflow-hidden rounded-md border border-primary-foreground/20 bg-primary-foreground/10 hover:bg-primary-foreground/15 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-foreground/70"
                  >
                    <img
                      src={attachment.source}
                      alt={attachment.label}
                      loading="lazy"
                      className="h-full w-full object-cover"
                    />
                  </button>
                ) : (
                  <div className="inline-flex max-w-[220px] items-center gap-1.5 rounded-md border border-primary-foreground/20 bg-primary-foreground/10 px-2 py-1 text-xs text-primary-foreground/90">
                    {attachment.kind === "image" ? (
                      <ImageIcon className="h-3.5 w-3.5 shrink-0" />
                    ) : (
                      <Film className="h-3.5 w-3.5 shrink-0" />
                    )}
                    <span className="truncate">{attachment.label}</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="flex h-7 items-center gap-0.5 text-[11.5px] text-[var(--text-subtle)] opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
        <MessageTime timestamp={message.created_at} />
        {text.trim() ? <CopyButton text={text} label="Copy message" /> : null}
      </div>
    </article>
  );
}
