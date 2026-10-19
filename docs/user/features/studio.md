---
title: "Studio"
description: "Narrate long scripts paragraph by paragraph, then export them as one audio file."
icon: "layout-dashboard"
---
Studio turns long scripts (chapters, articles, podcast scripts) into narrated
audio. Izwi splits a script into paragraphs that you can edit, listen to and
re-render one at a time, then export as one file.

Open **Studio** from the sidebar. Each project opens at `/studio/:projectId`.

## Create a project

1. Select **New project** (or press ⌘N / Ctrl+N).
2. Paste your script, or select **Import a file** for a `.txt`, `.md` or
   `.html` file.
3. Choose a model and a voice. Select the play button next to a voice to hear
   it first.
4. Select **Create project**.

If the model isn't downloaded yet, Studio offers to download it. Paragraphs
render once it is on this device.

---

## Edit the script

A project reads like a document: one block per paragraph.

| To | Do this |
|----|---------|
| Change wording | Click into a paragraph and type. Edits save on their own. |
| Split a paragraph | Put the cursor where it should break and press **Enter**. |
| Join two paragraphs | Press **Backspace** at the start of the second one. |
| Add a paragraph | Hover between two paragraphs and select **+**, or select **Add paragraph** at the end. |
| Reorder | Drag the handle to the left of a paragraph, or use **Move up** / **Move down** in its **⋯** menu. |
| Delete | Select **Delete** in the paragraph's **⋯** menu. |

Shift-click or ⌘-click (Ctrl-click) a paragraph's play button to select it.
With paragraphs selected, a bar offers **Render**, **Export** and **Delete**.

---

## Render

Rendering runs on the Izwi server, one paragraph at a time. It keeps going
when you leave Studio, and picks up where it stopped after the server
restarts.

The button at the top right always shows the next step:

- **Render 12 paragraphs** for a new project.
- **Render 3 changes** after edits: only paragraphs whose audio is missing,
  out of date or failed.
- **Rendering 4 of 12**, with a stop button. Stopping finishes the paragraph
  in progress and leaves the rest.
- **Export** when every paragraph is current.

Each paragraph shows where it stands:

| State | Meaning |
|-------|---------|
| Play button and length | Rendered and current. |
| **Changed** | The text, voice, speed or pronunciation changed since it was rendered. Its old audio still plays until you render it again. |
| **Queued** / **Rendering…** | Waiting for, or in, the render queue. |
| **Failed** | It didn't render; the reason is shown under the paragraph. |

Select a paragraph's render button to render just that one, or **Render
again** in its **⋯** menu.

---

## Listen

The player at the bottom plays the whole project, paragraph after
paragraph, and highlights the one playing. Use it to skip between
paragraphs, scrub through the project and change the playback speed.

---

## Voice, speed and pronunciation

Select the voice button at the top of a project to open its settings:

- **Voice**: the model and voice every paragraph uses.
- **Speed**: for models that support it.
- **Pronunciation**: replace a word with how it should be said, for names,
  acronyms or product terms. Rules match whole words in any letter case.

Saving a new voice or speed marks the affected paragraphs as changed, so you
choose when to render them again.

To give one paragraph a different voice (a second speaker, say), open its
**⋯** menu and select **Voice for this paragraph**. Choose **Project voice**
to return it to the project's voice.

---

## Export

Select **Export** to save the project as one WAV file.

- **Pause between paragraphs** adds a short silence at each break.
- **Include script** saves the text as a `.txt` file next to the audio.

If any paragraph isn't rendered or has changed, the button reads **Render
and export**: Studio renders those paragraphs first, then saves the file.
Paragraphs rendered with different models are matched to one sample rate.

## Using the API

Studio's preview API is documented in the [API Reference](/api#studio).

---

## See Also

- [Text-to-Speech](/features/text-to-speech)
- [Voice Presets](/models/voice-presets)
- [API Reference](/api#studio)
