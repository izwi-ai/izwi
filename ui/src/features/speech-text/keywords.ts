/** Keywords typed one per line or separated by commas, whitespace collapsed
 * and duplicates dropped (the server applies the same normalization). */
export function parseKeywords(text: string): string[] {
  const keywords: string[] = [];
  for (const raw of text.split(/[\n,]/)) {
    const keyword = raw.trim().replace(/\s+/g, " ");
    if (keyword && !keywords.includes(keyword)) {
      keywords.push(keyword);
    }
  }
  return keywords;
}
