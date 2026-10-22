import { Link, useLocation } from "react-router";

import { PageState, pageStateButtonClass } from "@/components/ui/page-state";

export function NotFoundPage() {
  const location = useLocation();
  const requestedPath = `${location.pathname}${location.search}`;

  return (
    <PageState
      align="center"
      title="Page not found"
      description={
        <>
          Nothing lives at{" "}
          <code className="break-all font-mono text-[13px] text-(--text-secondary)">
            {requestedPath}
          </code>
          .
        </>
      }
      actions={
        <>
          <Link to="/voice" className={pageStateButtonClass.primary}>
            Open Voice
          </Link>
          <Link to="/transcription" className={pageStateButtonClass.secondary}>
            Open Transcription
          </Link>
          <Link to="/models" className={pageStateButtonClass.secondary}>
            Open Models
          </Link>
        </>
      }
    />
  );
}
