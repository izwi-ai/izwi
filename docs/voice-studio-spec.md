# Voice Studio Specification

## Goal

Unify voice creation and management workflows under a single `/voice-studio` route while preserving existing capabilities and avoiding workflow regressions.

## Scope

- Consolidate these legacy surfaces into one studio experience:
  - `/voices`
  - `/voice-cloning`
  - `/voice-design`
- Keep all existing model requirements, generation flows, saved voice actions, and TTS handoff behavior.
- Keep styling and interaction patterns aligned with the existing Izwi design system in both light and dark themes.

## Information Architecture

`/voice-studio` owns all voice workflows and is split into three top-level tabs:

1. `Library`
- Browse and manage saved voices.
- Preview and use built-in voices.
- Search/filter by provenance (`Cloned`, `Designed`) and metadata.

2. `Clone`
- Clone a voice from reference audio + transcript.
- Save cloned outputs to reusable saved voices.
- Route to Text-to-Speech with selected saved voice.

3. `Design`
- Generate voice candidates from natural-language description.
- Compare generated candidates.
- Save designed outputs to reusable saved voices.
- Route to Text-to-Speech with selected saved voice.

## URL Contract

- Primary route: `/voice-studio`
- Tab query contract: `/voice-studio?tab=library|clone|design`
- Invalid/missing `tab` query falls back to `library`.

## Navigation & Migration

- Sidebar creation section should expose a single `Voice Studio` nav item.
- Legacy routes remain as redirects:
  - `/voices` -> `/voice-studio?tab=library`
  - `/voice-cloning` -> `/voice-studio?tab=clone`
  - `/voice-design` -> `/voice-studio?tab=design`
- Redirects should preserve deep-link usability and reduce migration friction.

## UX Requirements

- Keep one page shell and one page header for studio context.
- Maintain current route typography tokens and shared workspace components.
- Reuse existing model-selection modal patterns per workflow.
- Preserve route-level history affordances for clone/design workflows.
- Keep all actions available with no feature loss versus legacy routes.

## Non-Goals

- No backend schema/API changes.
- No saved voice provenance format changes (`voice_cloning`, `voice_design`).
- No redesign outside the current design system.

## Rollout & Verification

- Deliver in incremental commits:
  1. Spec and planning artifact.
  2. Route and navigation foundation with redirects.
  3. Library integration in studio tab.
  4. Clone/design integration in studio tabs.
  5. Tests and verification sweep.
- Verify with focused UI tests and typecheck before final sign-off.
