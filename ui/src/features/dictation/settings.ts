import type {
  DictationSettings,
  DictationSettingsPatch,
} from "@/features/dictation/types";

/** The settings with a change applied; `cleanup` merges field by field. */
export function mergeDictationSettings(
  current: DictationSettings,
  patch: DictationSettingsPatch,
): DictationSettings {
  const { cleanup, ...rest } = patch;
  return {
    ...current,
    ...rest,
    cleanup: cleanup ? { ...current.cleanup, ...cleanup } : current.cleanup,
  };
}

/** A patch that puts back what `patch` changed, read from `previous`. */
export function revertPatch(
  previous: DictationSettings,
  patch: DictationSettingsPatch,
): DictationSettingsPatch {
  const revert: Record<string, unknown> = {};
  for (const key of Object.keys(patch) as (keyof DictationSettings)[]) {
    if (key === "cleanup") {
      const fields = Object.keys(patch.cleanup ?? {}) as Array<
        keyof DictationSettings["cleanup"]
      >;
      revert.cleanup = Object.fromEntries(
        fields.map((field) => [field, previous.cleanup[field]]),
      );
    } else {
      revert[key] = previous[key];
    }
  }
  return revert as DictationSettingsPatch;
}

/** Terms trimmed, blanks dropped, repeats (ignoring case) removed. */
export function normalizeVocabulary(terms: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const term of terms) {
    const trimmed = term.trim();
    const key = trimmed.toLowerCase();
    if (trimmed && !seen.has(key)) {
      seen.add(key);
      result.push(trimmed);
    }
  }
  return result;
}
