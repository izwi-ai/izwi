import { type ReactNode, useRef } from "react";
import clsx from "clsx";
import ReactMarkdown, { type ExtraProps } from "react-markdown";
import remarkGfm from "remark-gfm";

import { CopyButton } from "@/components/ui/CopyButton";

interface MarkdownContentProps {
  content: string;
  className?: string;
}

type Element = NonNullable<ExtraProps["node"]>;

function supportsRegExpLookbehind(): boolean {
  try {
    new RegExp("(?<=a)b");
    return true;
  } catch {
    return false;
  }
}

/**
 * GFM (tables, strikethrough, autolinks) where the engine can run it.
 * remark-gfm's autolinks use a regex lookbehind, which WebKit before
 * Safari 16.4 lacks (macOS 12, early macOS 13): there it threw on every
 * reply, so Markdown renders without GFM instead.
 */
export function remarkPluginsFor(supportsLookbehind: boolean) {
  return supportsLookbehind ? [remarkGfm] : [];
}

const REMARK_PLUGINS = remarkPluginsFor(supportsRegExpLookbehind());

function isExternalLink(href?: string): boolean {
  if (!href) {
    return false;
  }
  return /^(https?:)?\/\//i.test(href);
}

/** Language named on a fenced block (```rust), read from its code element. */
function codeLanguage(pre: Element | undefined): string | null {
  const code = pre?.children.find(
    (child): child is Element =>
      child.type === "element" && child.tagName === "code",
  );
  const classNames = code?.properties?.className;
  if (!Array.isArray(classNames)) {
    return null;
  }
  for (const name of classNames) {
    if (typeof name === "string" && name.startsWith("language-")) {
      return name.slice("language-".length) || null;
    }
  }
  return null;
}

function CodeBlock({
  language,
  children,
}: {
  language: string | null;
  children: ReactNode;
}) {
  const preRef = useRef<HTMLPreElement | null>(null);
  return (
    <div className="chat-code-block">
      <div className="chat-code-block-bar">
        <span>{language ?? "code"}</span>
        <CopyButton
          text={() => preRef.current?.textContent ?? ""}
          label="Copy"
          showLabel
        />
      </div>
      <pre ref={preRef}>{children}</pre>
    </div>
  );
}

export function MarkdownContent({ content, className }: MarkdownContentProps) {
  return (
    <div className={clsx("chat-markdown", className)}>
      <ReactMarkdown
        remarkPlugins={REMARK_PLUGINS}
        components={{
          a: ({ href, children, node: _node, ...props }) => {
            const external = isExternalLink(href);
            return (
              <a
                {...props}
                href={href}
                target={external ? "_blank" : undefined}
                rel={external ? "noopener noreferrer nofollow" : undefined}
              >
                {children}
              </a>
            );
          },
          pre: ({ node, children }) => (
            <CodeBlock language={codeLanguage(node)}>{children}</CodeBlock>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
