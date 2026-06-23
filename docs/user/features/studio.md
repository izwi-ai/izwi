---
title: "Studio"
description: "Manage long-form text-to-speech projects, chapter workflows, and exports in Izwi Studio."
icon: "layout-dashboard"
---
# Studio

Studio is Izwi's long-form text-to-speech workspace for single-device, solo production workflows.

Open **Studio** from the sidebar, or open a specific project at
`/studio/:projectId`.

## What You Can Do

- Create reusable narration projects from pasted text, local files, or URL imports.
- Add, split, merge, reorder, edit, and bulk-manage script segments.
- Assign shared project settings (model, voice, speed, folder) for consistent output.
- Queue background renders with retry/cancel and automatic resume after reload.
- Apply pronunciation replacement rules before segment synthesis.
- Export full projects or selected segments with output presets and optional script sidecar files.
- Save snapshots and restore earlier project states when an edit path goes wrong.

## Typical Workflow

1. Create a project and import script content.
2. Review and refine segment structure.
3. Configure project profile and pronunciation rules.
4. Queue renders for changed or pending segments.
5. Verify queue completion and restore from snapshots if needed.
6. Export audio (and optional script sidecar).

---

## Projects and Folders

Projects hold the script, segment list, render state, and shared generation
settings. Folders let you organize related projects without changing how
rendering works.

Project metadata can include:

- Title
- Folder
- Model
- Voice or saved voice
- Speed
- Export preferences

When you change shared project settings, new renders use the updated profile.
Previously rendered audio stays available until you re-render affected segments.

---

## Importing and Segmenting Text

Studio can create projects from pasted text, local files, or URL imports. After
import, review the segment structure before rendering.

Common segment actions:

| Action | Use it to |
|--------|-----------|
| Edit | Fix wording or punctuation for one segment. |
| Split | Break a long segment into smaller render units. |
| Merge | Combine neighboring segments that should render together. |
| Reorder | Move segments into the final listening order. |
| Bulk delete | Remove multiple unwanted segments. |

Shorter, well-punctuated segments are easier to re-render and review.

---

## Pronunciation Rules

Pronunciation rules are project-level replacements applied before segment
synthesis. Use them for names, acronyms, product terms, or words that a selected
voice consistently mispronounces.

Review pronunciation rules before large renders; changing a rule affects future
renders and any segment you choose to re-render.

---

## Render Queue

Studio renders segment audio in the background. The queue supports:

- Rendering all pending or changed segments
- Rendering selected segments
- Retrying failed segments
- Cancelling queued or active render work
- Resuming after page reload

Each segment keeps its own render state, so a failed segment does not require
rerendering the whole project.

---

## Snapshots

Snapshots capture earlier project states. Use them before major restructuring,
bulk edits, or large imports.

Restore a snapshot when an edit path goes wrong. After restore, review pending
render state because restored text/settings may require fresh segment audio.

---

## Exporting

Studio can export:

- The full rendered project
- Selected segments
- Audio with output presets
- Optional script sidecar files

Exports use rendered segment audio. If segments are missing or stale, render or
re-render them first.

## Using the API

Studio's preview API is documented in the [API Reference](/api#studio).
It covers projects, folders, metadata, pronunciations, snapshots, render jobs,
segment editing, and project audio export query parameters.

---

## See Also

- [Text-to-Speech](/features/text-to-speech)
- [Voice Presets](/models/voice-presets)
- [API Reference](/api#studio)
