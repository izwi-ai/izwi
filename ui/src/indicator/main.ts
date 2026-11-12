import { invoke, isTauri } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

import type {
  DictationOs,
  DictationSnapshot,
  DictationStatus,
} from "@/features/dictation/types";
import {
  HIDDEN_VIEW,
  nextIndicatorView,
  pushLevel,
  type IndicatorView,
} from "@/indicator/view";

// The page the desktop app shows in its small always-on-top window while
// dictating. It only draws the latest status: the app shows and hides the
// window itself, and nothing here can be clicked.

const BAR_COUNT = 5;
const REDUCED_MOTION_LEVEL_INTERVAL_MS = 250;

const ICON_ATTRIBUTES =
  'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"';

const ICONS = {
  mic: `<svg ${ICON_ATTRIBUTES}><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 11a7 7 0 0 0 14 0"/><path d="M12 18v4"/></svg>`,
  lock: `<svg ${ICON_ATTRIBUTES}><rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>`,
  spinner: `<svg ${ICON_ATTRIBUTES}><path d="M21 12a9 9 0 1 1-6.2-8.6"/></svg>`,
  check: `<svg ${ICON_ATTRIBUTES}><path d="M20 6 9 17l-5-5"/></svg>`,
};

function element(className: string, html = ""): HTMLSpanElement {
  const node = document.createElement("span");
  node.className = className;
  node.innerHTML = html;
  return node;
}

function guessOs(): DictationOs {
  const platform = navigator.platform || navigator.userAgent;
  if (/mac/i.test(platform)) {
    return "macos";
  }
  return /win/i.test(platform) ? "windows" : "linux";
}

function start(root: HTMLElement) {
  const pill = document.createElement("div");
  pill.className = "pill";
  const bars = element("bars", "<i></i>".repeat(BAR_COUNT));
  const text = element("text");
  const words = document.createElement("bdi");
  text.append(words);
  pill.append(
    element("mic", ICONS.mic),
    bars,
    element("lock", ICONS.lock),
    element("spinner", ICONS.spinner),
    element("check", ICONS.check),
    text,
  );
  root.append(pill);

  const barNodes = Array.from(bars.children) as HTMLElement[];
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  let os = guessOs();
  let view: IndicatorView = HIDDEN_VIEW;
  let partial = "";
  let levels: number[] = [];
  let lastLevelAt = 0;
  let sawStatus = false;

  const drawBars = () => {
    barNodes.forEach((bar, index) => {
      const level = levels[levels.length - BAR_COUNT + index] ?? 0;
      bar.style.transform = `scaleY(${(0.15 + 0.85 * level).toFixed(3)})`;
    });
  };

  const draw = () => {
    const shown =
      view.kind === "notice"
        ? view.text
        : view.kind === "listening"
          ? partial
          : "";
    pill.dataset.kind = view.kind;
    pill.dataset.locked = String(view.kind === "listening" && view.locked);
    pill.dataset.tone = view.kind === "notice" ? view.tone : "neutral";
    pill.dataset.hasText = String(shown.length > 0);
    words.textContent = shown;
  };

  const applyStatus = (status: DictationStatus) => {
    sawStatus = true;
    if (status.state === "starting") {
      partial = "";
      levels = [];
      drawBars();
    }
    view = nextIndicatorView(view, status, os);
    draw();
  };

  draw();

  if (!isTauri()) {
    return;
  }

  void listen<DictationStatus>("dictation://status", (event) =>
    applyStatus(event.payload),
  );
  void listen<string>("dictation://partial", (event) => {
    partial = event.payload.replace(/\s+/g, " ").trim();
    draw();
  });
  void listen<number>("dictation://level", (event) => {
    const now = performance.now();
    if (
      reducedMotion.matches &&
      now - lastLevelAt < REDUCED_MOTION_LEVEL_INTERVAL_MS
    ) {
      return;
    }
    lastLevelAt = now;
    levels = pushLevel(levels, event.payload, BAR_COUNT);
    drawBars();
  });

  // The window may open mid-dictation: start from where things stand.
  void invoke<DictationSnapshot>("dictation_snapshot")
    .then((snapshot) => {
      os = snapshot.platform.os;
      if (!sawStatus) {
        applyStatus(snapshot.status);
      }
    })
    .catch(() => undefined);
}

const root = document.getElementById("indicator");
if (root) {
  start(root);
}
