// Formats the server decodes (symphonia: WAV, MP3, AAC/M4A, FLAC, Ogg
// Vorbis). Opus is not among them, so .opus and .webm files are not offered.
const AUDIO_FILE_EXTENSIONS = [
  ".wav",
  ".mp3",
  ".m4a",
  ".aac",
  ".flac",
  ".ogg",
] as const;
const AUDIO_MIME_TYPES = [
  "audio/wav",
  "audio/x-wav",
  "audio/mpeg",
  "audio/mp4",
  "audio/aac",
  "audio/flac",
  "audio/x-flac",
  "audio/ogg",
] as const;

// Explicit extensions keep the macOS/Tauri file picker selectable. WebKit does
// not reliably translate the audio/* wildcard into native allowed-file types.
export const AUDIO_FILE_ACCEPT = [
  ...AUDIO_FILE_EXTENSIONS,
  ...AUDIO_MIME_TYPES,
].join(",");

/** The one line every audio picker shows under its drop zone. */
export const AUDIO_FILE_HINT = "WAV, MP3, M4A, AAC, FLAC or OGG";
