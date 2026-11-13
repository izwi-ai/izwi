---
title: "Dictation"
description: "Hold a shortcut in any app, speak, and Izwi types what you said at the cursor. Runs on your computer."
sidebarTitle: "Dictation"
icon: "mic"
---
Dictation turns speech into text wherever your cursor is: an email, a chat box, a code editor, a terminal. Hold a shortcut, speak, let go, and the text appears. The speech model runs on your computer, and no audio is saved.

Dictation is part of the desktop app on macOS, Windows and Linux.

---

## Getting started

1. Open **Dictation** in the sidebar and turn it on.
2. Follow **Setup**:
   - **Speech model**: download `Parakeet-TDT-0.6B-v3` if it is not installed. Izwi keeps the dictation model loaded for as long as dictation is on, so the first word is as fast as the rest.
   - **Keyboard control** (macOS only): allow Izwi under **System Settings › Privacy & Security › Accessibility**. Izwi needs it to paste for you.
   - Try it in the box on the page.
3. Click into any app, hold the shortcut, speak, and release.

| System | Default shortcut |
|--------|------------------|
| macOS | `⌥ Space` |
| Windows, Linux | `Ctrl+Shift+Space` |

Izwi must be running. Closing its window keeps it in the tray, and **Settings › Launch at login** starts it with your computer.

---

## Using it

| Action | How |
|--------|-----|
| Dictate | Hold the shortcut, speak, release |
| Hands-free | Tap the shortcut twice; press it once more to finish |
| Cancel | Press `Esc` while recording |
| Start or stop without the keyboard | Tray icon › **Start or Stop Dictation** |
| Insert the last text again | Tray icon › **Paste Last Dictation**, or its own shortcut |

Under **Shortcut › How it works** you can switch from holding the key to pressing it once to start and once to stop.

A small indicator near the bottom of the screen shows when Izwi is listening and when it is transcribing. A recording stops by itself after five minutes.

### Choosing a shortcut

A shortcut is one key with modifiers, such as `Ctrl+Shift+Space`, or a function key alone, such as `F9`. Click the shortcut box and press the keys, together or one after another. To use modifier keys alone, press them and wait a second.

On macOS and Windows it can also be modifier keys alone: hold `⌥`, `Ctrl+Shift`, or `Fn` (macOS). If you press another key while holding them, Izwi treats it as a normal keyboard shortcut and discards the recording. Modifier-only shortcuts need the Accessibility permission on macOS and stop working while a password field has focus; shortcuts with a key do not have that limit.

---

## How the text is inserted

By default Izwi puts the text on the clipboard, pastes it, and then puts your previous clipboard back. The dictated entry is marked so clipboard managers and Windows clipboard history skip it.

**Text › Insert text by** offers two alternatives:

| Option | Use it when |
|--------|-------------|
| **Typing** | An app blocks pasting. Slower for long text |
| **Clipboard only** | You want to paste yourself |

When Izwi cannot insert the text, it leaves it on the clipboard and the indicator says so. This happens when:

- Izwi has no Accessibility permission (macOS).
- The focused window runs as administrator (Windows). A normal app cannot send keys to it.
- The desktop offers no way to send keys (some Wayland desktops, see below).

On Windows only text is saved and restored around a paste: a copied image or file is replaced by the dictated text.

---

## Speech

| Setting | What it does |
|---------|--------------|
| **Speech model** | Any installed speech-to-text model. Parakeet is the fastest; pick another for languages it does not cover |
| **Language** | Leave on automatic unless the model picks the wrong language. Short phrases are the usual case: set your language and Parakeet keeps to its alphabet instead of, say, writing English in Cyrillic |
| **Microphone** | The system default, or a specific device |
| **Keep microphone ready** | Keeps the microphone open so the first word is never clipped. Your system shows its microphone indicator all the time, and a Bluetooth headset stays in call mode |

Typical wait between releasing the key and seeing text, measured with Parakeet on an Apple Silicon Mac: about half a second for ten seconds of speech. On computers without a supported GPU the model runs on the CPU and takes longer.

### Live preview

Models that transcribe while you speak (`Nemotron-3.5-ASR-Streaming-0.6B`, `Voxtral-Mini-4B-Realtime-2602`, `LFM3-Audio-2B-GGUF`) show the words in the indicator as you say them, and the wait after releasing the key stays short however long you spoke. Turn it off under **Feedback › Live preview**.

---

## Text

| Setting | What it does |
|---------|--------------|
| **Add a space after** | Ends the text with a space, so the next dictation does not run into it |
| **Remove filler words** | Drops "um", "uh" and similar |
| **Vocabulary** | Names and terms the speech model should expect |
| **Replacements** | Replaces a phrase with another, matched as whole words. `new line` → a line break is a common one |
| **Clean up with AI** | Passes the text through a small local chat model that fixes punctuation and removes false starts |

**Clean up with AI** adds a short delay (under a second with `Qwen3.5-2B` on Apple Silicon). If the model takes longer than 1.5 seconds, Izwi inserts the text as it was heard. Replacements are applied after the clean-up.

---

## History

Izwi keeps the text of your recent dictations on this computer so you can copy one again. Audio is never stored. Turn it off under **History › Keep history**, or clear it there.

---

## Linux

On **X11** dictation works as on other systems. Typing instead of pasting needs `xdotool`.

On **Wayland** the desktop controls shortcuts and key input:

| Desktop | Shortcut | Paste |
|---------|----------|-------|
| GNOME 48 or later, KDE Plasma | Registered through the desktop; it may ask you to confirm the key | The desktop asks once for permission to send keys |
| Hyprland, niri | Registered through the desktop portal | `wtype` |
| sway and others without a shortcuts portal | Bind a key yourself (below) | `wtype` |

If the Dictation page says the shortcut could not be registered, bind a key in your desktop's keyboard settings to:

```bash
izwi-desktop --dictation toggle
```

`start`, `stop` and `cancel` are also accepted. The command talks to the Izwi that is already running.

Without `wtype`, `ydotool` or `dotool`, and without the desktop's permission, Izwi leaves the text on the clipboard for you to paste. Modifier-only shortcuts are not available on Linux.

---

## Troubleshooting

| Problem | What to check |
|---------|---------------|
| Nothing happens when you press the shortcut | Dictation is on; Izwi is running; no other app uses the same shortcut (the page shows an error if registration failed) |
| "Didn't catch that" | The right microphone is selected and not muted; on macOS, Izwi is allowed under **Privacy & Security › Microphone** |
| The text is only copied, not inserted | See [How the text is inserted](#how-the-text-is-inserted) |
| macOS: the text is only copied although Izwi is switched on under **Accessibility** | The permission belongs to an older copy of the app. Remove Izwi from the list with the minus button, then allow it again from the Dictation page |
| The first words are missing | Turn on **Keep microphone ready** |
| Names come out wrong | Add them to **Vocabulary**, or add a **Replacement** |
| It pastes in a terminal as `^V` | Use **Insert text by › Typing** for that terminal |

Each dictation's timing is written to `izwi-desktop.log` in the app's log folder, without the text.
