const APP_NAME = "Izwi";

function withAppName(pageTitle: string): string {
  return `${pageTitle} · ${APP_NAME}`;
}

export function legacyDiarizationTarget(
  recordId: string | null | undefined,
  search: string,
): string {
  const searchParams = new URLSearchParams(search);

  if (recordId) {
    searchParams.delete("create");
    searchParams.set("mode", "diarization");
    const query = searchParams.toString();
    return `/transcription/${encodeURIComponent(recordId)}${query ? `?${query}` : ""}`;
  }

  searchParams.delete("mode");
  searchParams.delete("job_kind");
  searchParams.set("create", "diarization");
  const query = searchParams.toString();
  return `/transcription${query ? `?${query}` : ""}`;
}

export function documentTitleForLocation(
  pathname: string,
  search: string,
): string {
  const segments = pathname.split("/").filter(Boolean);
  const root = segments[0] ?? "";
  const isDetail = segments.length > 1;
  const searchParams = new URLSearchParams(search);

  switch (root) {
    case "":
    case "voice":
      return withAppName("Voice");
    case "chat":
      return withAppName("Chat");
    case "transcription":
      if (isDetail) {
        const mode = (
          searchParams.get("mode") ?? searchParams.get("job_kind") ?? ""
        ).toLowerCase();
        if (mode === "diarization") {
          return withAppName("Diarization Record");
        }
        if (mode === "speaker_attributed_asr" || mode === "saa") {
          return withAppName("Speaker Attributed ASR Record");
        }
        return withAppName("Transcription Record");
      }
      return searchParams.get("create") === "diarization"
        ? withAppName("New Diarization")
        : withAppName("Transcription");
    case "diarization":
      return withAppName(isDetail ? "Diarization Record" : "Diarization");
    case "text-to-speech":
      return withAppName(isDetail ? "Text-to-Speech Record" : "Text to Speech");
    case "studio":
      return withAppName(isDetail ? "Studio Project" : "Studio");
    case "voices":
    case "voice-cloning":
    case "voice-design":
      return withAppName("Voices");
    case "models":
    case "my-models":
      return withAppName("Models");
    case "settings":
      return withAppName("Settings");
    default:
      return withAppName("Page Not Found");
  }
}
