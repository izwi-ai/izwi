---
title: "Settings and Onboarding"
description: "Configure appearance, updates, analytics, desktop system behavior, and first-run model setup in Izwi."
sidebarTitle: "Settings"
icon: "settings"
---
# Settings and Onboarding

The Settings page contains device-level controls for the local app. Open
**Settings** from the bottom of the sidebar.

---

## Settings Sections

| Section | Controls |
|---------|----------|
| Appearance | Choose System, Light, or Dark theme. |
| Updates | View the current app version, check for updates, and open an available update prompt. |
| Privacy | Enable or disable anonymous analytics for this device. |
| System | Desktop-only tray icon and launch-at-login controls. |

---

## Appearance

Theme choices are local to the device:

- **System** follows the OS preference.
- **Light** forces the light theme.
- **Dark** forces the dark theme.

The app shows the currently resolved theme next to the selector.

---

## Updates

The update section shows:

- Current app version
- Last update check time
- Whether an update is available
- Last updater error, when one exists

Use **Check now** to run a manual update check. When an update is available,
use **View** to open the update prompt.

Some builds can disable updater support. In that case, Settings shows the
disable reason instead of update actions.

---

## Privacy and Analytics

Anonymous analytics are opt-in per device. When enabled, the app records product
usage events such as route views, model actions, and onboarding completion.

Prompts, transcripts, audio, and personal identifiers are not sent.

The preference is stored through:

| Route | Purpose |
|-------|---------|
| `GET /v1/preferences` | Read `{ analytics_opt_in }`. |
| `PUT /v1/preferences/analytics` | Update the analytics preference with `{ "opt_in": true }`. |

---

## Desktop System Controls

These controls are available only in the desktop app:

| Control | Effect |
|---------|--------|
| Show tray icon | When enabled, closing the window keeps Izwi running in the tray. When disabled, closing exits Izwi. |
| Launch at login | Opens Izwi automatically when you sign in. |

The browser UI shows these controls as unavailable because it cannot change OS
desktop integration settings.

---

## First-Run Onboarding

On first launch, Izwi shows a three-step onboarding flow:

1. Review the feature list.
2. Choose quick setup or custom setup.
3. Start using Izwi.

The model setup step groups downloads by:

- Chat
- Text to Speech
- Voice Studio: Cloning
- Voice Studio: Design
- Transcription
- Diarization

Quick setup picks a starter set. Custom setup lets you choose the exact model
categories to download. You can always adjust models later from
[Model Management](/models/management).

Onboarding state is stored through:

| Route | Purpose |
|-------|---------|
| `GET /v1/onboarding` | Read completion state and current analytics preference. |
| `POST /v1/onboarding/complete` | Mark onboarding complete. |

---

## See Also

- [Getting Started](/getting-started)
- [Model Management](/models/management)
- [API Reference](/api#onboarding-and-preferences)
