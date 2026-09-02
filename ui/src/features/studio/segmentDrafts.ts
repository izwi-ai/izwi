export interface StudioSegmentDraft {
  segmentId: string;
  baseText: string;
  draftText: string;
}

export interface StudioSegmentDraftConflict extends StudioSegmentDraft {
  serverText: string;
}

interface StudioSegmentText {
  id: string;
  text: string;
}

interface RestoredStudioSegmentDrafts {
  drafts: Record<string, string>;
  conflicts: Record<string, StudioSegmentDraftConflict>;
}

const STORAGE_KEY_PREFIX = "izwi.studio.segment-drafts.v1";

function storageKey(projectId: string): string {
  return `${STORAGE_KEY_PREFIX}:${encodeURIComponent(projectId)}`;
}

function sessionStorageOrNull(): Storage | null {
  if (typeof window === "undefined") {
    return null;
  }

  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

function isStudioSegmentDraft(value: unknown): value is StudioSegmentDraft {
  if (!value || typeof value !== "object") {
    return false;
  }

  const candidate = value as Partial<StudioSegmentDraft>;
  return (
    typeof candidate.segmentId === "string" &&
    candidate.segmentId.length > 0 &&
    typeof candidate.baseText === "string" &&
    typeof candidate.draftText === "string"
  );
}

export function readStudioSegmentDrafts(
  projectId: string,
  storage: Storage | null = sessionStorageOrNull(),
): StudioSegmentDraft[] {
  if (!storage || !projectId) {
    return [];
  }

  try {
    const raw = storage.getItem(storageKey(projectId));
    if (!raw) {
      return [];
    }
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }

    const draftsBySegmentId = new Map<string, StudioSegmentDraft>();
    for (const entry of parsed) {
      if (isStudioSegmentDraft(entry) && entry.baseText !== entry.draftText) {
        draftsBySegmentId.set(entry.segmentId, entry);
      }
    }
    return [...draftsBySegmentId.values()];
  } catch {
    return [];
  }
}

function writeStudioSegmentDrafts(
  projectId: string,
  drafts: StudioSegmentDraft[],
  storage: Storage | null,
): void {
  if (!storage || !projectId) {
    return;
  }

  try {
    if (drafts.length === 0) {
      storage.removeItem(storageKey(projectId));
      return;
    }
    storage.setItem(storageKey(projectId), JSON.stringify(drafts));
  } catch {
    // Session storage is best effort; editing must keep working when unavailable.
  }
}

export function storeStudioSegmentDraft(
  projectId: string,
  draft: StudioSegmentDraft,
  storage: Storage | null = sessionStorageOrNull(),
): void {
  if (!storage || !projectId || !draft.segmentId) {
    return;
  }

  const drafts = readStudioSegmentDrafts(projectId, storage).filter(
    (entry) => entry.segmentId !== draft.segmentId,
  );
  if (draft.baseText !== draft.draftText) {
    drafts.push(draft);
  }
  writeStudioSegmentDrafts(projectId, drafts, storage);
}

export function removeStudioSegmentDraft(
  projectId: string,
  segmentId: string,
  storage: Storage | null = sessionStorageOrNull(),
): void {
  if (!storage || !projectId || !segmentId) {
    return;
  }
  writeStudioSegmentDrafts(
    projectId,
    readStudioSegmentDrafts(projectId, storage).filter(
      (entry) => entry.segmentId !== segmentId,
    ),
    storage,
  );
}

export function removeStudioSegmentDrafts(
  projectId: string,
  segmentIds: Iterable<string>,
  storage: Storage | null = sessionStorageOrNull(),
): void {
  const removedIds = new Set(segmentIds);
  if (!storage || !projectId || removedIds.size === 0) {
    return;
  }
  writeStudioSegmentDrafts(
    projectId,
    readStudioSegmentDrafts(projectId, storage).filter(
      (entry) => !removedIds.has(entry.segmentId),
    ),
    storage,
  );
}

export function clearStudioProjectDrafts(
  projectId: string,
  storage: Storage | null = sessionStorageOrNull(),
): void {
  if (!storage || !projectId) {
    return;
  }
  try {
    storage.removeItem(storageKey(projectId));
  } catch {
    // Session storage is best effort.
  }
}

export function restoreStudioSegmentDrafts(
  projectId: string,
  segments: StudioSegmentText[],
  storage: Storage | null = sessionStorageOrNull(),
): RestoredStudioSegmentDrafts {
  const serverTextBySegmentId = new Map(
    segments.map((segment) => [segment.id, segment.text]),
  );
  const drafts: Record<string, string> = {};
  const conflicts: Record<string, StudioSegmentDraftConflict> = {};

  for (const draft of readStudioSegmentDrafts(projectId, storage)) {
    const serverText = serverTextBySegmentId.get(draft.segmentId);
    if (serverText === undefined) {
      continue;
    }
    if (serverText === draft.baseText) {
      drafts[draft.segmentId] = draft.draftText;
      continue;
    }
    conflicts[draft.segmentId] = { ...draft, serverText };
  }

  return { drafts, conflicts };
}
