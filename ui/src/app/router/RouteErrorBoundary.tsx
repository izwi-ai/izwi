import { Component, useState, type ErrorInfo, type ReactNode } from "react";
import { TriangleAlert } from "lucide-react";

import { PageState, pageStateButtonClass } from "@/components/ui/page-state";
import { APP_VERSION } from "@/shared/config/runtime";

interface RouteErrorBoundaryProps {
  children: ReactNode;
  resetKey?: string;
}

interface RouteErrorBoundaryState {
  error: Error | null;
}

function errorDetails(error: Error): string {
  return [
    `${error.name}: ${error.message}`,
    `Route: ${window.location.pathname}${window.location.search}`,
    `Izwi ${APP_VERSION}`,
  ].join("\n");
}

function RouteErrorFallback({
  error,
  onRetry,
  onReload,
}: {
  error: Error;
  onRetry: () => void;
  onReload: () => void;
}) {
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">(
    "idle",
  );
  const details = errorDetails(error);

  const copyDetails = async () => {
    try {
      await navigator.clipboard.writeText(details);
      setCopyState("copied");
    } catch {
      setCopyState("failed");
    }
  };

  return (
    <PageState
      role="alert"
      icon={TriangleAlert}
      tone="danger"
      title="This page hit a problem"
      description="Your recordings and history are safe. Reloading usually fixes it."
      actions={
        <>
          <button
            type="button"
            onClick={onReload}
            className={pageStateButtonClass.primary}
          >
            Reload page
          </button>
          <button
            type="button"
            onClick={onRetry}
            className={pageStateButtonClass.secondary}
          >
            Try again
          </button>
          <button
            type="button"
            onClick={() => void copyDetails()}
            className={pageStateButtonClass.secondary}
          >
            {copyState === "copied"
              ? "Copied"
              : copyState === "failed"
                ? "Couldn't copy"
                : "Copy details"}
          </button>
        </>
      }
    >
      <details open className="border-t border-[var(--border-muted)] pt-4">
        <summary className="cursor-pointer text-[13px] text-[var(--text-muted)]">
          Details
        </summary>
        <pre className="mt-3 whitespace-pre-wrap break-words rounded-[10px] border border-[var(--border-subtle)] bg-[var(--surface-panel)] p-3 font-mono text-[12px] leading-5 text-[var(--text-subtle)]">
          {details}
        </pre>
      </details>
    </PageState>
  );
}

export class RouteErrorBoundary extends Component<
  RouteErrorBoundaryProps,
  RouteErrorBoundaryState
> {
  state: RouteErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: unknown): RouteErrorBoundaryState {
    return {
      error: error instanceof Error ? error : new Error("Unknown route error"),
    };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Route failed to render:", error, info.componentStack);
  }

  componentDidUpdate(previousProps: RouteErrorBoundaryProps) {
    if (previousProps.resetKey !== this.props.resetKey && this.state.error) {
      // Recover on navigation without remounting a healthy route and its streams.
      this.setState({ error: null });
    }
  }

  private retry = () => {
    this.setState({ error: null });
  };

  private reload = () => {
    window.location.reload();
  };

  render() {
    if (!this.state.error) {
      return this.props.children;
    }

    return (
      <RouteErrorFallback
        error={this.state.error}
        onRetry={this.retry}
        onReload={this.reload}
      />
    );
  }
}
