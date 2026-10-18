import { Film, ImageIcon } from "lucide-react";

import type { ChatThreadMessageRecord } from "@/api";
import { MarkdownContent } from "@/components/ui/MarkdownContent";
import {
  type ImagePreviewState,
  parseUserMessageDisplayFromContentParts,
} from "@/features/chat/playground/support";
import { cn } from "@/lib/utils";

interface UserMessageProps {
  message: ChatThreadMessageRecord;
  onOpenImage: (preview: ImagePreviewState) => void;
}

/** A sent message: its text plus any image or video attachments. */
export function UserMessage({ message, onOpenImage }: UserMessageProps) {
  const parsed = parseUserMessageDisplayFromContentParts(message);

  return (
    <>
      {(!parsed || parsed.text || parsed.attachments.length === 0) && (
        <MarkdownContent
          content={
            parsed && parsed.attachments.length > 0
              ? parsed.text
              : message.content
          }
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
    </>
  );
}
