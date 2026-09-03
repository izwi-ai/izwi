import { Link, useLocation } from "react-router-dom";

import { PageHeader, PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";

export function NotFoundPage() {
  const location = useLocation();
  const requestedPath = `${location.pathname}${location.search}`;

  return (
    <PageShell>
      <PageHeader
        title="Page not found"
        description="This link does not match an Izwi workspace. Choose where you want to continue."
      />
      <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-6 sm:p-8">
        <p className="text-sm text-[var(--text-muted)]">
          Requested path:{" "}
          <code className="break-all rounded bg-[var(--bg-surface-2)] px-1.5 py-1 text-[var(--text-secondary)]">
            {requestedPath}
          </code>
        </p>
        <div className="mt-5 flex flex-wrap gap-3">
          <Button asChild>
            <Link to="/voice">Open Voice</Link>
          </Button>
          <Button asChild variant="outline">
            <Link to="/transcription">Open Transcription</Link>
          </Button>
          <Button asChild variant="outline">
            <Link to="/models">Open Models</Link>
          </Button>
        </div>
      </div>
    </PageShell>
  );
}
