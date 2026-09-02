# UI browser certification

The Playwright suite is the deterministic UI boundary. It exercises the web
application with intercepted API responses at the supported 1280×720,
960×680, and 390×844 viewports.

Run it from `ui/`:

```bash
npm run test:e2e
```

The accessibility matrix runs serious/critical WCAG axe checks in both light
and dark themes at every configured viewport:

```bash
npm run test:e2e -- e2e/accessibility.spec.ts
```

On failure, Playwright writes screenshots and retained traces under
`ui/test-results/`. CI also writes the HTML report under
`ui/playwright-report/`.

## Packaged desktop boundary

The repository does not currently provide a packaged Tauri/WebDriver harness.
Browser tests therefore do not certify updater installation, tray routing, or
launch-at-login behavior, because mocking those APIs in Chromium would give a
false desktop signal. Those boundaries remain covered by Rust tests and the
release-OS packaged evidence pass until a dedicated Tauri driver is added.
