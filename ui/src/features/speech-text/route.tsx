import { useMemo } from "react";
import { useSearchParams } from "react-router-dom";

import type { SharedPageProps } from "@/app/router/types";
import { DiarizationPage } from "@/features/diarization/route";
import { TranscriptionPage } from "@/features/transcription/route";

function resolveSpeechTextMode(
  searchParams: URLSearchParams,
): "transcription" | "diarization" {
  const mode = searchParams.get("mode")?.trim().toLowerCase();
  if (mode === "diarization") {
    return "diarization";
  }

  const jobKind = searchParams.get("job_kind")?.trim().toLowerCase();
  if (jobKind === "diarization") {
    return "diarization";
  }

  return "transcription";
}

export function SpeechTextPage(props: SharedPageProps) {
  const [searchParams] = useSearchParams();
  const mode = useMemo(
    () => resolveSpeechTextMode(searchParams),
    [searchParams],
  );

  if (mode === "diarization") {
    return <DiarizationPage {...props} routeBasePath="/transcription" />;
  }

  return <TranscriptionPage {...props} />;
}
