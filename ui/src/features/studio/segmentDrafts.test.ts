import { beforeEach, describe, expect, it } from "vitest";

import {
  clearStudioProjectDrafts,
  readStudioSegmentDrafts,
  removeStudioSegmentDraft,
  restoreStudioSegmentDrafts,
  storeStudioSegmentDraft,
} from "@/features/studio/segmentDrafts";

describe("Studio segment draft storage", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  it("isolates drafts by project and restores matching text after navigation or reload", () => {
    storeStudioSegmentDraft("project-a", {
      segmentId: "segment-1",
      baseText: "Server A",
      draftText: "Draft A",
    });
    storeStudioSegmentDraft("project-b", {
      segmentId: "segment-1",
      baseText: "Server B",
      draftText: "Draft B",
    });

    expect(
      restoreStudioSegmentDrafts("project-a", [
        { id: "segment-1", text: "Server A" },
      ]).drafts,
    ).toEqual({ "segment-1": "Draft A" });
    expect(
      restoreStudioSegmentDrafts("project-b", [
        { id: "segment-1", text: "Server B" },
      ]).drafts,
    ).toEqual({ "segment-1": "Draft B" });
  });

  it("clears only the saved or deleted segment entry", () => {
    storeStudioSegmentDraft("project-a", {
      segmentId: "segment-1",
      baseText: "One",
      draftText: "Edited one",
    });
    storeStudioSegmentDraft("project-a", {
      segmentId: "segment-2",
      baseText: "Two",
      draftText: "Edited two",
    });

    removeStudioSegmentDraft("project-a", "segment-1");

    expect(readStudioSegmentDrafts("project-a")).toEqual([
      {
        segmentId: "segment-2",
        baseText: "Two",
        draftText: "Edited two",
      },
    ]);
  });

  it("does not restore stale text when the server segment changed", () => {
    storeStudioSegmentDraft("project-a", {
      segmentId: "segment-1",
      baseText: "Original server text",
      draftText: "Unsaved local edit",
    });

    const restored = restoreStudioSegmentDrafts("project-a", [
      { id: "segment-1", text: "Newer server text" },
    ]);

    expect(restored.drafts).toEqual({});
    expect(restored.conflicts["segment-1"]).toEqual({
      segmentId: "segment-1",
      baseText: "Original server text",
      draftText: "Unsaved local edit",
      serverText: "Newer server text",
    });
    expect(readStudioSegmentDrafts("project-a")).toHaveLength(1);
  });

  it("removes all drafts only after an explicit project clear", () => {
    storeStudioSegmentDraft("project-a", {
      segmentId: "segment-1",
      baseText: "One",
      draftText: "Edited one",
    });
    storeStudioSegmentDraft("project-b", {
      segmentId: "segment-1",
      baseText: "Two",
      draftText: "Edited two",
    });

    clearStudioProjectDrafts("project-a");

    expect(readStudioSegmentDrafts("project-a")).toEqual([]);
    expect(readStudioSegmentDrafts("project-b")).toHaveLength(1);
  });
});
