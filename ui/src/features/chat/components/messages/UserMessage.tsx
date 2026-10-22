import { useEffect, useRef, useState } from "react";
import { Film, ImageIcon, Pencil } from "lucide-react";

import type { ChatThreadMessageRecord } from "@/api";
import { Button } from "@/components/ui/button";
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
  /** Offer Edit: a text message while no turn is running. */
  editable: boolean;
  editing: boolean;
  onEdit: () => void;
  onCancelEdit: () => void;
  onSubmitEdit: (text: string) => void;
}

const actionButtonClass =
  "inline-flex h-7 w-7 items-center justify-center rounded-lg text-(--text-subtle) transition-colors hover:bg-(--surface-raised) hover:text-(--text-primary) focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-(--border-strong)";

/** Edit-in-place for a sent message. Enter sends, Escape cancels. */
function EditMessageForm({
  initialText,
  onCancel,
  onSubmit,
}: {
  initialText: string;
  onCancel: () => void;
  onSubmit: (text: string) => void;
}) {
  const [draft, setDraft] = useState(initialText);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
  }, []);

  const submit = () => {
    if (draft.trim()) {
      onSubmit(draft);
    }
  };

  return (
    <form
      className="ml-[18%] flex flex-col gap-1.5 self-stretch rounded-2xl border border-(--border-strong) bg-(--surface-panel) px-3 py-2.5"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
    >
      <textarea
        ref={textareaRef}
        aria-label="Edit your message"
        value={draft}
        rows={Math.min(8, Math.max(2, draft.split("\n").length))}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault();
            submit();
          } else if (event.key === "Escape") {
            event.preventDefault();
            onCancel();
          }
        }}
        className="w-full resize-none bg-transparent text-[0.9375rem] leading-relaxed text-(--text-primary) focus:outline-hidden"
      />
      <div className="flex flex-wrap items-center gap-2 text-xs text-(--text-subtle)">
        <span className="min-w-0 flex-1">
          Sending replaces the replies below this message.
        </span>
        <Button type="button" variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit" size="sm" disabled={!draft.trim()}>
          Send
        </Button>
      </div>
    </form>
  );
}

/** A sent message in its bubble, with its time, Copy and Edit shown on
 * hover or focus. */
export function UserMessage({
  message,
  onOpenImage,
  editable,
  editing,
  onEdit,
  onCancelEdit,
  onSubmitEdit,
}: UserMessageProps) {
  const parsed = parseUserMessageDisplayFromContentParts(message);
  const text =
    parsed && parsed.attachments.length > 0 ? parsed.text : message.content;

  if (editing) {
    return (
      <article aria-label="Editing your message" className="flex flex-col">
        <EditMessageForm
          initialText={message.content}
          onCancel={onCancelEdit}
          onSubmit={onSubmitEdit}
        />
      </article>
    );
  }

  return (
    <article
      aria-label="Your message"
      className="group flex max-w-[78%] flex-col items-end gap-1 self-end"
    >
      <div className="wrap-break-word rounded-[18px] bg-(--surface-raised) px-4 py-2.5 text-[0.9375rem] leading-relaxed text-(--text-primary)">
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
                    className="block h-16 w-16 overflow-hidden rounded-md border border-primary-foreground/20 bg-primary-foreground/10 hover:bg-primary-foreground/15 transition-colors focus:outline-hidden focus:ring-2 focus:ring-primary-foreground/70"
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
      <div className="flex h-7 items-center gap-0.5 text-[11.5px] text-(--text-subtle) opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
        <MessageTime timestamp={message.created_at} />
        {text.trim() ? <CopyButton text={text} label="Copy message" /> : null}
        {editable ? (
          <button
            type="button"
            onClick={onEdit}
            aria-label="Edit message"
            title="Edit"
            className={actionButtonClass}
          >
            <Pencil className="h-[15px] w-[15px]" aria-hidden="true" />
          </button>
        ) : null}
      </div>
    </article>
  );
}
