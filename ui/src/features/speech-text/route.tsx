import { useEffect, useMemo } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";

import type { SharedPageProps } from "@/app/router/types";
import { DiarizationPage } from "@/features/diarization/route";
import { TranscriptionPage } from "@/features/transcription/route";

function resolveSpeechTextMode(
  searchParams: URLSearchParams,
  options?: { recordId?: string },
): "transcription" | "speaker_attributed_asr" | "diarization" {
  if (!options?.recordId) {
    return "transcription";
  }

  const mode = searchParams.get("mode")?.trim().toLowerCase();
  if (mode === "diarization") {
    return "diarization";
  }
  if (mode === "speaker_attributed_asr" || mode === "saa") {
    return "speaker_attributed_asr";
  }

  const jobKind = searchParams.get("job_kind")?.trim().toLowerCase();
  if (jobKind === "diarization") {
    return "diarization";
  }
  if (jobKind === "speaker_attributed_asr" || jobKind === "saa") {
    return "speaker_attributed_asr";
  }

  return "transcription";
}

export function SpeechTextPage(props: SharedPageProps) {
  const { recordId } = useParams<{ recordId: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  useEffect(() => {
    if (recordId) {
      return;
    }

    const nextSearchParams = new URLSearchParams(searchParams);
    const hadMode = nextSearchParams.has("mode");
    const hadJobKind = nextSearchParams.has("job_kind");
    if (hadMode) {
      nextSearchParams.delete("mode");
    }
    if (hadJobKind) {
      nextSearchParams.delete("job_kind");
    }
    if (!hadMode && !hadJobKind) {
      return;
    }

    const nextQuery = nextSearchParams.toString();
    navigate(nextQuery ? `/transcription?${nextQuery}` : "/transcription", {
      replace: true,
    });
  }, [navigate, recordId, searchParams]);

  const mode = useMemo(
    () => resolveSpeechTextMode(searchParams, { recordId }),
    [recordId, searchParams],
  );

  if (mode === "diarization") {
    return <DiarizationPage {...props} routeBasePath="/transcription" />;
  }
  if (mode === "speaker_attributed_asr") {
    return (
      <TranscriptionPage {...props} speechTextMode="speaker_attributed_asr" />
    );
  }

  return <TranscriptionPage {...props} />;
}
