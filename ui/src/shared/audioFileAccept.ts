const AUDIO_FILE_EXTENSIONS = [".wav", ".mp3", ".m4a", ".aac"] as const;
const AUDIO_MIME_TYPES = [
  "audio/wav",
  "audio/x-wav",
  "audio/mpeg",
  "audio/mp4",
  "audio/aac",
] as const;

// Explicit extensions keep the macOS/Tauri file picker selectable. WebKit does
// not reliably translate the audio/* wildcard into native allowed-file types.
export const AUDIO_FILE_ACCEPT = [
  ...AUDIO_FILE_EXTENSIONS,
  ...AUDIO_MIME_TYPES,
].join(",");

