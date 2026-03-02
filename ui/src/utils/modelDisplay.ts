export function withQwen3Prefix(name: string, variant: string): string {
  const trimmed = name.trim();
  const normalizedVariant = variant.toLowerCase();
  const qwenPrefix = normalizedVariant.startsWith("qwen3.5")
    ? "Qwen3.5"
    : normalizedVariant.startsWith("qwen3")
      ? "Qwen3"
      : null;

  if (!qwenPrefix) {
    return trimmed || variant;
  }

  if (!trimmed) {
    return qwenPrefix;
  }

  if (/^qwen3(\.5)?\b/i.test(trimmed)) {
    return trimmed;
  }

  return `${qwenPrefix} ${trimmed}`;
}
