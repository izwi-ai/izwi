import { Component, type ErrorInfo, type ReactNode } from "react";
import { TriangleAlert } from "lucide-react";

import { Button } from "@/components/ui/button";
import { StatePanel } from "@/components/ui/state-panel";

interface RouteErrorBoundaryProps {
  children: ReactNode;
}

interface RouteErrorBoundaryState {
  error: Error | null;
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
      <div role="alert" className="flex min-h-[20rem] items-center justify-center">
        <StatePanel
          className="w-full max-w-2xl"
          align="center"
          tone="danger"
          icon={TriangleAlert}
          eyebrow="Page error"
          title="This page could not be opened"
          description={
            <>
              Your navigation is still available. Try opening the page again, or
              reload Izwi if the problem continues.
              <span className="mt-2 block text-xs">{this.state.error.message}</span>
            </>
          }
          actions={
            <>
              <Button type="button" size="sm" onClick={this.retry}>
                Try again
              </Button>
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={this.reload}
              >
                Reload Izwi
              </Button>
            </>
          }
        />
      </div>
    );
  }
}
