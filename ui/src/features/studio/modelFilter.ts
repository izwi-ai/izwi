import type { ModelInfo } from "@/api";

/**
 * Studio projects render with a built-in speaker or a saved voice only; the
 * server rejects other voice modes, so description-only models (VoiceDesign)
 * and tokenizers are not offered.
 */
export function isStudioRenderableModel(
  model: Pick<ModelInfo, "variant" | "speech_capabilities"> | undefined,
): boolean {
  const capabilities = model?.speech_capabilities;
  if (!model || !capabilities || model.variant.includes("Tokenizer")) {
    return false;
  }
  return (
    capabilities.supports_builtin_voices || capabilities.supports_reference_voice
  );
}
