export type SpeakerAccent = {
  solid: string;
  soft: string;
  border: string;
};

/**
 * A speaker's colour, by the speaker's position in its transcript (talk-time
 * rank on diarization records, first appearance on speaker-attributed ASR),
 * so the same position gets the same colour on both pages.
 */
export function speakerAccent(index: number): SpeakerAccent {
  const hue = (index * 67 + 154) % 360;
  return {
    solid: `hsl(${hue} 72% 52%)`,
    soft: `hsla(${hue} 72% 52% / 0.16)`,
    border: `hsla(${hue} 72% 52% / 0.4)`,
  };
}
