/** Formats Izwi encodes saved speech into on download (`?format=`). */
export type AudioDownloadFormat = "wav" | "mp3" | "flac" | "opus";

export const AUDIO_DOWNLOAD_FORMATS: { id: AudioDownloadFormat; label: string }[] = [
  { id: "wav", label: "WAV" },
  { id: "mp3", label: "MP3" },
  { id: "flac", label: "FLAC" },
  { id: "opus", label: "Opus" },
];

export const audioDownloadFormatLabel = (format: AudioDownloadFormat) =>
  AUDIO_DOWNLOAD_FORMATS.find((option) => option.id === format)?.label ?? "WAV";
