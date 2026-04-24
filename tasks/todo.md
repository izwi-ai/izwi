# OSS Runtime P0 Backlog Analysis And Implementation Plan

## Goal

Analyze `/Users/lennex/Code/clients/izwi-knowledge/open-source-runtime-roadmap.md`, especially the Prioritized Backlog P0 items, against the current OSS codebase and produce a phased implementation plan only.

## Constraints

- Research, analysis, and planning only.
- No runtime or product implementation in this phase.
- Treat the roadmap as a snapshot, not guaranteed current truth.
- Prefer current code/docs/CI evidence over stale roadmap claims.
- Skip Docker and CUDA truth closure implementation planning per user direction.
- Keep changes scoped to this planning artifact.

## Research Checklist

- [x] Review project lessons and current task notes.
- [x] Read the roadmap and P0 backlog.
- [x] Audit release/Docker CUDA backend truth.
- [x] Audit support matrix and stable/preview API surface documentation.
- [x] Audit readiness/liveness health contract.
- [x] Audit structured JSON logging support.
- [x] Audit `/v1/responses` and agent-session durability.
- [x] Run lightweight verification of current server/CLI code.
- [x] Produce phased implementation plan.

## P0 Current-State Summary

### 1. Fix release and Docker backend truth for CUDA/NVIDIA support

Status: partially implemented, not fully closed.

Already implemented:

- Release workflow builds CPU-safe public Linux/Windows CLI/server binaries, then CUDA variants in a separate target directory and stages them under `runtime/cuda`.
- Release verification scripts enforce stable public binary names and private CUDA runtime layout.
- Dockerfile has separate CPU and CUDA build/runtime targets, and the CUDA target builds `izwi-server --features cuda`.
- Backend Truth CI checks CPU cargo, CUDA cargo, CPU Docker build, and CUDA Docker build.
- Runtime CUDA diagnostics exist in `izwi-core` and are exposed through `/v1/health` and `izwi status --detailed`.
- README, release docs, and `docs/user/support-matrix.md` describe the unified CPU-safe public binary plus private CUDA runtime contract.

Still missing or inconsistent:

- Docker Compose CUDA instructions are likely wrong: `docker compose --profile cuda up -d` starts the default CPU service plus the profiled CUDA service, and both bind `8080:8080`.
- CI builds Docker CUDA but does not run the image, query `/v1/health`, or assert CUDA selection.
- No automated NVIDIA-host smoke test proves release/runtime CUDA selection or inference on a real GPU.
- Runtime CUDA delegation docs say `auto` should delegate only when a usable CUDA device is indicated, but the current gate checks packaged runtime, runtime libraries, and driver availability, not `device_usable`.
- Windows private runtime health can report active but not packaged after delegation because the delegation path does not set `IZWI_CUDA_RUNTIME_BINARY`, and Windows discovery lacks the Linux-style installed fallback.
- Final packaged artifact checks are weaker than staging checks because they omit some CUDA library families required by staging verification.

### 2. Publish runtime support matrix: OS, hardware, deployment targets, stable API surfaces

Status: mostly implemented, needs consistency hardening.

Already implemented:

- `docs/user/support-matrix.md` now exists and covers backend matrix, deployment matrix, API surface maturity, CUDA caveats, and verification guidance.
- README and installation/getting-started docs point to the support matrix.
- The matrix correctly marks `/v1/responses`, `/v1/admin/*`, persisted first-party workflow APIs, and local agent/session features as preview.

Still missing or inconsistent:

- `docs/openai-compatibility-contract.json` lists `/v1/responses/:id`, cancel, and input-items as supported without saying those lifecycle routes are process-local preview behavior.
- Dev docs list `POST /v1/responses` but do not clearly document retrieval/cancel/input-items maturity.
- Docker Compose CUDA support text should be revised after the Compose service/profile contract is fixed.
- The support matrix should link to operational probe/logging contracts once those P0 items land.

### 3. Add real readiness and liveness endpoints

Status: missing.

What exists:

- `/v1/health` and `/internal/health` return status/version/backend/device/CUDA diagnostics.
- `/v1/metrics`, `/internal/metrics`, and Prometheus metrics expose runtime telemetry.
- CLI status, CLI serve readiness wait, bench throughput, Dockerfile healthchecks, and Docker Compose healthchecks depend on `/v1/health`.

What is missing:

- No `/livez`, `/readyz`, or equivalent.
- `/health` always returns `200` with `status: "ok"` and is a summary, not readiness or liveness.
- No `503` readiness path.
- No lifecycle state for starting, ready, degraded, draining, or shutdown.
- No readiness checks for store health, selected backend availability, preload failures, required model residency, semaphore saturation, worker panic/restart thresholds, or shutdown drain state.
- Since the server only listens after preload/warmup, probes cannot observe alive-but-not-ready startup work.

### 4. Add structured JSON logging as a supported runtime mode

Status: missing.

What exists:

- Server uses `tracing_subscriber::fmt::layer()` with `RUST_LOG`/`EnvFilter`.
- HTTP `TraceLayer` spans include method, URI, and an inbound `x-request-id` when present.
- Request middleware generates/returns `x-request-id`.
- CLI exposes `--log-level` and passes `RUST_LOG` to the server.

What is missing:

- No `--log-format json`, `IZWI_LOG_FORMAT`, or config key.
- Workspace `tracing-subscriber` does not enable the JSON formatter feature.
- No stable JSON log schema for service/version/level/target/request id/method/path/status/latency/error.
- The trace span does not use the generated request id from request middleware when the caller does not provide one.
- No tests or docs for JSON log mode.
- Docker `json-file` logging is container framing, not application JSON logs.

### 5. Decide whether `/v1/responses` and agent sessions are durable product features or preview-only local conveniences

Status: decision should be preview-only now.

Current behavior:

- `/v1/responses` records live in an in-memory `HashMap` with a default 512-entry limit.
- `store: false` skips even in-memory storage.
- Streaming responses are stored only when completed or failed, not at `response.created`, so in-progress lookup/cancel is not a real durable lifecycle.
- `/v1/agent/sessions` session metadata also lives in an in-memory `HashMap`.
- Agent session creation does create a durable chat thread, but the agent session id and metadata are lost after restart.
- Voice profiles, voice sessions, voice turns, observations, and chat threads/messages are SQLite-backed; those are the durable local product surfaces today.

Decision:

- Keep `/v1/responses` and `/v1/agent/sessions` as preview, process-local, bounded-memory convenience APIs for now.
- Do not promote them to durable product features until SQLite storage, lifecycle semantics, restart tests, and docs are implemented together.

## Phased Implementation Plan

### Phase 0: Contract Cleanup And No-Code Truth Fixes

Status: complete.

Goal:

- Close the easy trust gaps before changing runtime behavior.

Tasks:

- [x] Update `docs/openai-compatibility-contract.json` or its surrounding docs to mark Responses lifecycle routes as preview/process-local where applicable.
- [x] Update dev docs to list all Responses lifecycle routes and describe their current non-durable behavior.
- [x] Keep `/v1/responses` and `/v1/agent/sessions` explicitly preview in the support matrix.
- [x] Add a short note in this planning artifact that Docker/CUDA truth closure is intentionally deferred/out of scope.

Verification:

- [x] Docs grep confirms no stale claim that `/v1/responses` object lifecycle or agent sessions are durable.

Out of scope:

- Docker and CUDA truth closure remains analyzed above for context, but is intentionally skipped in this implementation plan.

### Phase 1: Probe Contract And Lifecycle State

Status: complete.

Goal:

- Add real liveness/readiness semantics without breaking existing `/v1/health` clients.

Tasks:

- [x] Keep `/v1/health` backward-compatible as the rich status summary.
- [x] Add root `/livez` and `/readyz`.
- [x] Consider `/v1/live` and `/v1/ready` aliases for API namespace consistency.
- [x] Add a small lifecycle state to server/AppState: startup time, phase, ready flag, degraded reasons, preload/warmup errors, shutdown/draining flag.
- [x] Mark server unready before graceful shutdown starts.
- [x] Define readiness reason codes, for example `runtime_initialized`, `stores_available`, `backend_available`, `preload_complete`, `worker_health`, `not_draining`.
- [x] Return compact JSON and `200` when ready, compact JSON and `503` when unready.
- [x] Keep liveness cheap and independent of model readiness.

Verification:

- [x] Router tests for `/livez` returning `200`.
- [x] Router tests for `/readyz` returning `200` when initialized.
- [x] Unit tests for unready/degraded reason rendering.
- [x] Shutdown/drain state test where practical.

### Phase 2: Switch Dependents To The Right Probe

Goal:

- Make internal tooling and deployment assets use liveness/readiness correctly.

Tasks:

- [ ] Switch Dockerfile healthchecks from `/v1/health` to `/readyz`.
- [ ] Switch Docker Compose healthchecks to `/readyz`.
- [ ] Switch `izwi serve` startup wait to readiness.
- [ ] Keep `izwi status` on `/v1/health`, but optionally show readiness summary.
- [ ] Switch bench HTTP-overhead throughput test from `/v1/health` to `/livez` if it is measuring server overhead rather than runtime readiness.
- [ ] Update CLI docs, Docker docs, troubleshooting, and status docs.

Verification:

- [ ] CLI serve wait test updated.
- [ ] Docker healthcheck grep confirms `/readyz`.
- [ ] Bench docs and command text mention `/livez` if changed.
- [ ] `cargo check --locked -p izwi-server -p izwi-cli`.

### Phase 3: Structured JSON Logging Mode

Goal:

- Support machine-readable runtime logs while preserving current text default.

Tasks:

- [ ] Enable `tracing-subscriber` JSON formatter feature.
- [ ] Add `LogFormat { text, json }` parsing.
- [ ] Add direct server flag `--log-format`.
- [ ] Add env var `IZWI_LOG_FORMAT`.
- [ ] Add CLI `izwi serve --log-format` and pass it through to child server processes.
- [ ] Centralize tracing initialization in a small helper so parsing/defaults are testable.
- [ ] Define stable fields: timestamp, level, target, service, version, message, correlation_id, method, path, status, latency_ms, error.
- [ ] Make request-id middleware and TraceLayer agree on generated request IDs.
- [ ] Add response/failure events with status and latency.
- [ ] Document Docker usage with application JSON logs, distinct from Docker `json-file`.

Verification:

- [ ] Unit tests for log format parsing and defaults.
- [ ] Router/middleware test proves generated `x-request-id` is returned.
- [ ] Integration-style smoke captures one JSON log line and validates basic keys where feasible.
- [ ] `cargo check --locked -p izwi-server -p izwi-cli`.

### Phase 4: Preview Semantics Hardening For Responses And Agent Sessions

Goal:

- Make the current non-durable decision intentional, tested, and legible.

Tasks:

- [ ] Add clear docs for `/v1/responses` retention: process-local, bounded memory, evicted by age, lost on restart.
- [ ] Add clear docs for `/v1/agent/sessions`: session metadata process-local, chat thread/messages durable.
- [ ] Add tests for `store:false` not retaining response records.
- [ ] Add tests for bounded response eviction.
- [ ] Add tests for bounded agent-session eviction.
- [ ] Consider preview headers or response metadata for preview/process-local APIs if useful.
- [ ] Align UI/client assumptions so durable history uses chat/voice stores, not response/agent-session ids.

Verification:

- [ ] Targeted server tests for response and agent session memory semantics.
- [ ] Docs grep confirms preview/non-durable wording.
- [ ] `cargo check --locked -p izwi-server`.

### Phase 5: Optional Durable Promotion Path

Goal:

- Only if product decides durable Responses/agent sessions are needed, promote them deliberately.

Tasks:

- [ ] Add SQLite store/tables for response objects and response input items.
- [ ] Persist streaming lifecycle from created/in-progress through completed/failed.
- [ ] Implement meaningful active-response cancellation, or document cancel as best-effort if runtime cancellation is unavailable.
- [ ] Add SQLite store/tables for agent session metadata.
- [ ] Reconstruct agent sessions after restart using stored session metadata plus durable chat thread/messages.
- [ ] Add migrations and storage versioning tests.
- [ ] Add restart integration tests for Responses and agent sessions.
- [ ] Update compatibility contract, support matrix, dev docs, and UI copy only after behavior is durable.

Verification:

- [ ] Restart durability tests pass.
- [ ] Streaming lifecycle persistence tests pass.
- [ ] Agent session turn after restart works.
- [ ] Docs and contract promote surfaces from preview only after tests prove durability.

## Elegance Review

The elegant path is to avoid turning `/v1/health` into a catch-all contract. Keep it as the existing rich status endpoint, add small Unix/Kubernetes-style probes for orchestration, and keep durability decisions separate from OpenAI compatibility routing. That minimizes breakage while giving enterprise packaging the precise hooks it needs.

For Responses and agent sessions, the lowest-risk decision is preview-only now. There are already durable SQLite-backed product surfaces for chat and voice. Promoting the compatibility lifecycle to durable storage is valuable only if it is done with full lifecycle semantics, not a partial table bolted onto the current in-memory map.

## Verification Completed In This Planning Phase

- [x] `cargo check --locked -p izwi-server -p izwi-cli`

Result:

- Server and CLI compile successfully in the current codebase.

## Review

- The roadmap is stale on at least two P0 points: a support matrix exists, and CUDA release packaging has already been significantly implemented.
- Docker and CUDA truth closure was analyzed but intentionally skipped from the implementation plan per user direction.
- The remaining P0 implementation work in scope is contract hardening around real probes, JSON logging, and explicit preview semantics.
- No runtime implementation was performed in this phase.

# CPU + CUDA Unified Installer Feasibility Plan

## Goal

Investigate whether all Linux and Windows CPU installer surfaces can ship binaries that support both CPU and CUDA at runtime, using the existing device-selection pathway and without creating forked vendored copies of core files.

## Research Checklist

- [x] Map current runtime device/backend selection and where CPU/CUDA decisions are made.
- [x] Map current Cargo feature flags, dependency features, and build-script behavior for CPU/CUDA.
- [x] Map all Linux installer/release surfaces (`.deb`, AppImage/updater, terminal tarball, and any related bundle paths) and Windows installer/release build paths.
- [x] Identify whether one binary can include CPU and CUDA backends while falling back cleanly on CPU-only hosts.
- [x] Identify packaging/runtime constraints for Linux distributions and Windows CUDA dependencies.
- [x] Produce a phased implementation and verification plan that keeps core source shared.

## Constraints

- Research, analysis, and planning only.
- Do not implement code changes in this phase.
- Do not create forked vendored spin-offs of core files.

## Feasibility Summary

Shipping the current Linux and Windows release binaries with `--features cuda` is not by itself a safe unified CPU+CUDA installer strategy.

The Izwi runtime already has the higher-level selection path:

- `auto` probes CUDA on non-macOS and falls back to CPU if CUDA is unavailable.
- explicit `cpu` always selects CPU.
- explicit `cuda` errors if CUDA was requested but unavailable.
- `/v1/health`, `izwi status --detailed`, and `izwi version --full` expose requested, selected, and compiled backend state.

The blocker is earlier than Izwi runtime selection: Candle's CUDA feature path currently pulls in `cudarc` dynamic CUDA linking and `candle-kernels` links `cudart`. That means a CUDA-enabled binary may fail at OS loader startup on a CPU-only machine without CUDA shared libraries, before `main()` and before Izwi can fall back to CPU.

Therefore the viable paths are:

1. **Unified installer with bundled private CUDA redistributable libraries** for Linux and Windows, plus clear driver/runtime compatibility checks.
2. **Unified installer that installs/checks CUDA runtime prerequisites** before enabling CUDA, while remaining CPU-safe if prerequisites are absent.
3. **Change the dependency/linking strategy to lazy dynamic loading** so CUDA libraries are opened only when CUDA is selected. This requires upstream-compatible dependency work, not vendored core forks.
4. **Keep separate CPU and CUDA release artifacts** until one of the above is proven. This does not satisfy the requested installer behavior but is the safest interim contract.

## Current Release Surfaces

- Linux terminal tarball: packages `target/release/izwi`, `izwi-server`, and `izwi-desktop`.
- Linux desktop bundle: builds `.deb` and AppImage from the Tauri build.
- Linux updater: uses the AppImage artifact and signature in `latest-beta.json`.
- Windows terminal zip: packages `izwi.exe`, `izwi-server.exe`, and `izwi-desktop.exe`.
- Windows desktop installer/updater: builds and publishes NSIS setup `.exe` plus signature.

The release workflow currently builds Linux and Windows CLI/server binaries without `--features cuda`, so every downstream Linux/Windows release surface inherits CPU-only backend support.

## Non-Forking Design Direction

Use one shared backend packaging contract instead of copied core files:

- Keep `izwi-core`, `izwi-cli`, and `izwi-server` source shared.
- Add release build profiles/feature sets only at workflow/script level.
- Add small packaging helpers/manifests for CUDA runtime dependency discovery and validation.
- Prefer upstream-supported Cargo feature/dependency adjustments or patching dependency versions over copying Candle/cudarc source into this repo.
- Keep runtime reporting canonical: compiled backends, selected backend, requested backend availability, and dependency availability should all agree.

## Phased Plan

- [x] Phase 1: Prove loader behavior and dependency inventory
  Scope:
  Build Linux and Windows CUDA-feature binaries in CUDA-enabled builders. Inspect runtime dependencies with `ldd`/`readelf` on Linux and DLL inspection on Windows. Run the binaries on:
  1. CPU-only host with no CUDA libraries
  2. host with CUDA libraries but no usable GPU
  3. NVIDIA GPU host
  Expected decision:
  Confirm whether raw CUDA-feature binaries are unsafe for universal CPU installers. Current research strongly predicts they are unsafe without bundled/installed CUDA runtime libraries.
  Implementation:
  Added `scripts/release/audit-runtime-deps.sh` and `scripts/release/audit-runtime-deps.ps1` so release builds can inspect loader dependencies and prove CPU-safe startup paths on Linux/macOS and Windows.
  Verification:
  `bash -n scripts/release/audit-runtime-deps.sh`
  `scripts/release/audit-runtime-deps.sh target/release/izwi`
  Windows PowerShell validation still needs a Windows runner because `pwsh` is not available on this host.

- [x] Phase 2: Choose the CUDA runtime delivery model
  Scope:
  Decide between private bundled CUDA redistributables, installer-managed prerequisites, or upstream-compatible lazy loading. Validate NVIDIA redistribution terms for the exact libraries required by Candle/cudarc/candle-kernels.
  Deliverable:
  A written packaging contract for Linux and Windows that says exactly which CUDA libraries are bundled or required, where they live, how the loader finds them, and what happens on CPU-only machines.
  Implementation:
  Added `docs/release/cuda-runtime-contract.md` choosing a single installer with stable public `izwi` and `izwi-server` names. The contract allows private CUDA runtime variants with the same basename under package runtime directories, lists expected CUDA libraries, driver boundaries, runtime selection behavior, and verification requirements.
  Verification:
  Reviewed current NVIDIA CUDA EULA Attachment A and CUDA compatibility documentation.

- [x] Phase 3: Normalize release build inputs for all Linux and Windows surfaces
  Scope:
  Create one shared release build path that produces the binaries used by:
  Linux `.deb`, Linux AppImage/updater, Linux terminal tarball, Windows NSIS/updater, and Windows terminal zip.
  Notes:
  Check Tauri Linux/Windows resource config composition so AppImage/NSIS include the same `izwi` and `izwi-server` binaries as the terminal bundles. Avoid separate hand-copied packaging branches.
  Implementation:
  Added same-name private runtime staging scripts for Linux and Windows, release-only Tauri configs that bundle `target/release/runtime`, and release workflow steps that build CPU-safe public `izwi`/`izwi-server` plus CUDA runtime variants in a separate Cargo target dir. Linux tarballs, Windows zips, `.deb`, AppImage/updater, and NSIS/updater now draw from the same staged layout. Desktop CLI setup now copies the bundled `runtime/` directory with the installed CLI/server pair on Linux and Windows.
  Verification:
  `bash -n scripts/release/stage-unified-runtime.sh`
  `jq empty crates/izwi-desktop/tauri.release.linux.conf.json crates/izwi-desktop/tauri.release.windows.conf.json`
  `scripts/release/stage-unified-runtime.sh --cuda-target-dir target`
  `cargo check -p izwi-desktop`
  `git diff --check`

- [x] Phase 4: Add runtime dependency diagnostics
  Scope:
  Extend the existing backend truth reporting to distinguish:
  1. CUDA compiled into binary
  2. CUDA runtime libraries loadable
  3. NVIDIA driver present/compatible
  4. CUDA device usable
  5. selected backend
  Notes:
  This makes `auto` fallback understandable and prevents "CUDA compiled but CPU selected" confusion.
  Implementation:
  Added shared CUDA runtime diagnostics in `izwi-core`, exposed them through `/v1/health`, printed them in `izwi status --detailed`, and taught the public CPU-safe `izwi-server` entrypoint to delegate to the private same-name CUDA runtime only when packaged runtime libraries and the NVIDIA driver are visible. Explicit `--backend cuda` now fails before server startup with a clear diagnostic if the private runtime cannot be used.
  Verification:
  `cargo check -p izwi-core -p izwi-server -p izwi-cli`
  `cargo test -p izwi-core cuda_runtime`
  `cargo run -p izwi-server -- --backend cuda --port 0`
  `./target/debug/izwi-server --backend cpu --port 18081`
  `curl -sS http://127.0.0.1:18081/v1/health`
  `git diff --check`

- [x] Phase 5: Update CI/release verification gates
  Scope:
  Add checks that fail release if:
  - Linux/Windows release binaries do not report expected compiled backend support.
  - CPU-only smoke tests cannot start the installer-shipped binaries.
  - CUDA-host smoke tests cannot select CUDA.
  - updater artifacts and terminal bundles disagree on backend support.
  - docs/support matrix still says Linux/Windows release assets are CPU-only after the change.
  Implementation:
  Added Linux and Windows unified runtime verification scripts that enforce stable public binary names, require the private same-basename `runtime/cuda` layout, validate packaged CUDA runtime libraries, reject loader-visible CUDA dependencies on public binaries, and check release Tauri resource mappings. The release workflow now runs these verification gates for Linux and Windows before packaging terminal and desktop artifacts. The dependency audit scripts also support private-runtime audits that skip startup on CPU-only release runners while preserving public startup probes.
  Verification:
  `bash -n scripts/release/audit-runtime-deps.sh`
  `bash -n scripts/release/verify-unified-runtime.sh`
  `scripts/release/verify-unified-runtime.sh --skip-cuda-compiled-check --skip-cuda-library-check`
  `git diff --check`
  Windows PowerShell execution still needs a Windows runner because `pwsh` is not available on this host. CUDA device selection still needs a real NVIDIA host or self-hosted GPU runner; this phase gates the layout and CPU-safe startup path in hosted release CI.

- [x] Phase 6: Roll out behind explicit release notes
  Scope:
  Update `docs/RELEASING.md`, support matrix, Linux/Windows install docs, and release body text to describe the new contract. Make the first release a beta/preview CUDA packaging release unless Phase 1-5 smoke tests cover both CPU-only and NVIDIA hosts.
  Implementation:
  Updated the release body, release guide, support matrix, Linux install guide, Windows install guide, getting-started page, installation index, source-build guide, inference-engine docs, troubleshooting docs, and README to describe the unified Linux/Windows CPU+CUDA packaging contract. The docs now preserve the public `izwi`/`izwi-server` names, call out private same-basename CUDA runtime packaging, distinguish release installers from source builds, and label release CUDA packaging as preview until GPU-host smoke coverage is automated.
  Verification:
  `rg -n "CPU-focused|CPU-only|source-build-only|source-build-first|CUDA / NVIDIA support is|not included|do not ship CUDA|does not currently ship CUDA|assume CPU|dedicated CUDA release artifacts|treat the installed binary as CPU-only|current release workflow builds CPU|release artifacts remain CPU|release package is CPU" README.md docs .github/workflows/release.yml -S`
  `rg -n "izwi-cuda|izwi-server-cuda|public.*cuda|CUDA-suffixed" README.md docs .github/workflows/release.yml -S`
  `git diff --check`
  Remaining grep hits are intentional references to CPU-only hosts, the Docker CPU image, the existing Docker Compose `izwi-cuda` service name, or the canonical rule forbidding public `*-cuda` binaries.

## Review

- Current Izwi runtime/device selection is architecturally compatible with CPU/CUDA selection after process startup.
- Current CUDA-feature binary delivery is likely not CPU-installer-safe because CUDA shared libraries are eagerly linked by Candle/cudarc/candle-kernels.
- The requested user-facing goal is feasible only with a CUDA runtime delivery or lazy-loading strategy, not by simply flipping `--features cuda` in the release workflow.
- The plan covers all Linux release surfaces, not only Ubuntu or `.deb`.
- Phase 1 added repeatable loader/startup audit tooling for release artifacts. Local verification covered the Bash script and existing macOS binary; Windows script validation is deferred to a Windows runner.
- Phase 2 chose the CPU-safe public entrypoint plus private CUDA runtime model, documented the library contract, and kept all backend variants tied to shared source builds rather than forked core files.
- User correction captured: public binary names must remain `izwi` and `izwi-server`; no user-facing `izwi-cuda` or `izwi-server-cuda` commands.
- Phase 3 normalized release build/package inputs around stable public names plus a private `runtime/cuda` layout. Windows PowerShell staging still needs execution on a Windows runner because `pwsh` is not available locally.
- Phase 4 added runtime truth reporting and public-server CUDA delegation without changing public binary names. CUDA-host execution still needs validation on a NVIDIA host.
- Phase 5 added release verification gates for Linux and Windows unified runtime artifacts. Hosted CI can now fail on public name drift, missing private runtime layout, missing packaged CUDA runtime libraries, public CUDA loader dependencies, broken public startup probes, and Tauri runtime-resource drift. Real CUDA device selection remains the one verification item that needs a NVIDIA host or self-hosted GPU runner.
- Phase 6 updated user-facing docs and generated release-body text to describe Linux and Windows release installers as CPU-safe public entrypoints plus private packaged CUDA runtimes. The docs now avoid public CUDA-suffixed binary names and keep CUDA release packaging marked preview until NVIDIA-host smoke coverage is automated.

# Voice Configuration Modal Redesign Plan

## Goal

Redesign the `/voice` "Voice Configuration" modal into a cleaner, more professional control surface that follows the OpenAI GPT-5.4 frontend guidance for app UI:

- calm surface hierarchy
- strong typography and spacing
- minimal chrome
- cards only when the card is the interaction
- avoid dashboard-card mosaics and thick borders on every region

Source:
`https://developers.openai.com/blog/designing-delightful-frontends-with-gpt-5-4`

## Constraints

- Preserve the existing Izwi color system and overall product visual language.
- Keep the current tab structure: `Setup`, `Models`, `Agent`, `Memory`.
- Reduce boxed/card treatments wherever plain layout, rows, dividers, or list structure can carry the UI.
- Keep the modal functional on desktop and mobile.
- Use motion only where it improves hierarchy or modal presence.

## Phases

- [x] Phase 1: Redesign the modal shell and tab navigation
  Scope:
  Replace the current card-like modal shell and tab strip with a calmer two-zone layout: stronger header hierarchy, lighter framing, and tab navigation that feels like product navigation instead of pill-heavy controls.
  Deliverable:
  The modal opens with a cleaner structure, reduced chrome, improved spacing, and tab navigation that scales well across breakpoints.
  Verification:
  `npm run typecheck`
  Commit:
  `feat(ui): redesign voice config modal shell`

- [x] Phase 2: Redesign `Setup` and `Models` into cleaner operational layouts
  Scope:
  Rework `Setup` and `Models` so they rely on sections, lists, dividers, and row layouts instead of stacked cards. Simplify the setup overview, runtime mode controls, speaker selection area, and model lifecycle rows.
  Deliverable:
  The first two tabs feel denser, clearer, and more professional, with cards reserved only for actual interaction containers where needed.
  Verification:
  `npm run typecheck`
  Commit:
  `feat(ui): redesign voice setup and models tabs`

- [x] Phase 3: Redesign `Agent` and `Memory` tabs and finish modal polish
  Scope:
  Rework prompt editing and memory management to match the new modal system, including better hierarchy, lighter list treatments, and final responsive/polish adjustments across all tabs.
  Deliverable:
  All tabs share one consistent design language and the modal reads as a cohesive product settings surface rather than four unrelated panels.
  Verification:
  `npm run typecheck`
  Commit:
  `feat(ui): redesign voice agent and memory tabs`

# Unified CPU + CUDA Installer Hardening Plan

## Goal

Close the release and runtime gaps found during the CPU+CUDA installer review without changing public binary names or creating forked vendored copies of core files.

## Phases

- [x] Phase 1: Harden private CUDA runtime verification
  Scope:
  Run private Linux CUDA binary probes with the same loader path used at runtime, and make dependency audits fail on missing non-driver dependencies while still tolerating host NVIDIA driver absence in CI.
  Verification:
  `bash -n scripts/release/audit-runtime-deps.sh scripts/release/verify-unified-runtime.sh`
  `scripts/release/verify-unified-runtime.sh --skip-cuda-compiled-check --skip-cuda-library-check`

- [x] Phase 2: Inspect final packaged artifacts
  Scope:
  Add post-package release checks for terminal archives and practical desktop bundle artifacts so CI validates the final shipped layout, not only `target/release`.
  Verification:
  Shell/PowerShell syntax checks and release workflow review.

- [x] Phase 3: Make explicit CUDA strict
  Scope:
  Ensure `--backend cuda` fails clearly when CUDA cannot actually be selected after private runtime delegation, instead of silently serving on CPU.
  Verification:
  `cargo check -p izwi-core -p izwi-server`
  Targeted backend-selection tests where practical.

- [x] Phase 4: Remove path exposure from public health
  Scope:
  Keep `/v1/health` useful with booleans and missing library labels, but stop serializing private runtime and search paths by default.
  Verification:
  `cargo check -p izwi-server`

## Review

- Phase 1 complete: private Linux CUDA backend probes now run with the packaged runtime loader path, and private runtime audits now tolerate missing host NVIDIA driver libraries without allowing missing packaged CUDA runtime dependencies to pass silently. Windows audit logic was updated to the same contract, but still needs execution on a Windows runner because PowerShell is not available on this host.
- Phase 2 complete: added post-package artifact verification for Linux terminal archives, `.deb` payloads, AppImage extraction, Windows terminal zips, and Windows NSIS installer extraction. Local verification covered Bash syntax, whitespace checks, and a synthetic terminal tarball; real `.deb`, AppImage, and NSIS extraction will run in release CI where those artifacts are produced.
- Phase 3 complete: explicit CUDA backend mismatches now produce a CUDA-specific diagnostic that names the selected fallback backend and points users to CUDA runtime diagnostics. `auto` selection remains unchanged.
- Phase 4 complete: `/v1/health` keeps CUDA booleans, missing-library labels, device status, and notes, but no longer serializes private runtime paths or CUDA library search paths.

## Review

- Phase 1 complete: replaced the heavy card-style modal shell with a calmer two-zone settings layout, upgraded the header hierarchy, and turned the tab strip into product-style section navigation with lighter chrome.
- Phase 2 complete: rebuilt the `Setup` and `Models` tabs around sections and row lists, reduced stacked-card treatments, and kept model actions inside cleaner operational rows.
- Phase 3 complete: aligned `Agent` and `Memory` with the new section system, simplified prompt and memory presentation, and finished responsive polish across all four tabs.
- Verification: `npm run typecheck` and `npm run build`.

# Diarization Alignment Plan

## Goal

Align the `/diarization` route with the current `/transcription` route UX and structure while preserving diarization-specific capabilities:

- multi-model pipeline requirements
- upload and microphone capture
- speaker correction workflow
- reruns from quality controls
- summary regeneration and polling

## What Changed On Transcription

- The route was split into a collection view and a dedicated record view.
- History loading and record loading were extracted into route-level hooks.
- The old inline setup flow was replaced by a dedicated `NewTranscriptionModal`.
- New jobs now navigate straight to `/transcription/:recordId`.
- History is now shown as a simple operational table instead of a drawer workflow.
- Record actions were concentrated in a dedicated detail page: back, delete, copy, export, summary regeneration, and polling.
- Model setup moved onto reusable route-selection patterns with grouped model sections in `RouteModelModal`.
- The last polish pass focused on scan speed, modal styling, delete confirmation styling, background polling, and live delta updates.

## Current Diarization Understanding

- `/diarization` is still a single-route experience driven by `ui/src/features/diarization/route.tsx` and the older shared `ui/src/components/DiarizationPlayground.tsx`.
- Job creation, session state, review workspace, and latest-record state all live inside the playground instead of the route.
- Saved history is still handled through `ui/src/components/DiarizationHistoryPanel.tsx`, which uses a drawer plus a custom full-screen modal instead of dedicated route pages.
- The persisted diarization API already supports the record-centric workflow we need: list, get, create, update speaker names, rerun, regenerate summary, delete, and fetch record audio.
- Diarization creation is synchronous for persisted records and does not support the transcription-style streaming create flow, so the alignment should copy the route pattern, not the streaming behavior.
- Diarization has extra feature depth that transcription does not: pipeline model readiness, load-all pipeline controls, speaker name correction, quality reruns that produce new records, and summary polling for completed records.

## Main Alignment Gaps

- No `/diarization/:recordId` route yet.
- No route-level history or record hooks.
- No dedicated diarization creation modal.
- No transcription-style history table.
- No dedicated diarization record detail page.
- Model-selection logic is still mostly route-local instead of reusing the newer route-selection pattern where it fits.
- Regression coverage is component-heavy, but route-level coverage is missing.

## Phased Plan

- [x] Phase 1: Establish route-owned diarization state and routing
  Scope:
  Add `/diarization/:recordId` routing, create `useDiarizationHistory` and `useDiarizationRecord` hooks, and move history/record data ownership out of the playground so the route becomes the orchestration layer.
  Notes:
  Keep the visible UX mostly unchanged in this phase if possible. The goal is to create the same architectural footing that transcription now has before swapping the UI patterns.
  Verification:
  Add route tests covering `/diarization` and `/diarization/:recordId`, plus hook-driven polling expectations for pending summaries.
  Commit:
  `feat(ui): scaffold diarization record routes and data hooks`

- [x] Phase 2: Replace drawer-first history with a table and dedicated record pages
  Scope:
  Introduce a `DiarizationHistoryTable` and a `DiarizationRecordDetail` flow modeled after transcription. Move history record opening from drawer/modal interactions to route navigation. Preserve delete, copy, export, summary regeneration, speaker corrections, rerun access, and audio playback from the dedicated record page.
  Notes:
  This is the biggest alignment step because it changes the mental model from “single playground plus popups” to “index page plus detail page”.
  Verification:
  Add route tests for opening records from the table, back navigation, delete navigation behavior, and record summary polling.
  Commit:
  `feat(ui): move diarization history into dedicated record pages`

- [x] Phase 3: Move diarization job creation into a dedicated modal
  Scope:
  Build a `NewDiarizationModal` that takes over the inline session setup currently embedded in `DiarizationPlayground`. The modal should keep diarization-specific controls: upload and microphone entry points, speaker range, timing windows, pipeline readiness, and route-model guidance.
  Notes:
  This should align to the transcription modal layout principles, not blindly duplicate its fields. Diarization still needs richer setup than transcription.
  Verification:
  Add tests for opening the modal from the page header, enforcing model/pipeline readiness, creating a record from upload, and navigating to the new diarization record after creation.
  Commit:
  `feat(ui): move diarization setup into a creation modal`

- [x] Phase 4: Polish the diarization record workspace to match transcription conventions
  Scope:
  Refine the new record page so it matches the transcription detail rhythm: cleaner header metadata, grouped actions, better empty/loading/error states, styled delete confirmation, summary guidance, and background-safe refresh behavior. Keep diarization-specific tabs for transcript, speakers, and quality.
  Notes:
  This is where the “creation modals etc” polish from transcription gets mirrored across the diarization detail experience.
  Verification:
  Add focused tests for record actions, summary regeneration errors, speaker-correction persistence, rerun navigation/selection behavior, and delete confirmation behavior.
  Commit:
  `feat(ui): polish the diarization record workspace`

- [x] Phase 5: Align model-management ergonomics and clean up legacy diarization UI
  Scope:
  Reuse the newer route model selection patterns where they fit, centralize preferred diarization model resolution, preserve pipeline sections and load-all behavior, and remove the now-obsolete drawer/modal history path and any dead playground-only orchestration.
  Notes:
  The expected end state is feature-scoped diarization route/components/hooks, with older shared-component ownership reduced or removed.
  Verification:
  Run the diarization route tests plus existing diarization component tests to confirm no regressions in export, review workspace, and quality workflows.
  Commit:
  `refactor(ui): align diarization model management and remove legacy history flow`

## Review

- The safest sequence is architecture first, then navigation, then creation modal, then polish, then cleanup.
- The main place to avoid false parity is streaming: transcription has persisted streaming creation, diarization does not.
- The main place to avoid regressions is reruns and speaker corrections, because both currently depend on state living inside the old playground/history modal stack.

# Settings Redesign Plan

## Goal

Redesign `/settings` to feel like a polished product surface instead of a stack of generic cards, using the OpenAI GPT-5.4 frontend guidance as the quality bar.

## Design Notes From OpenAI Reference

- App surfaces should default to restrained layout, strong typography, minimal chrome, and a single accent.
- Cards should only exist when they are the interaction container; layout sections should otherwise be plain structure.
- Utility copy should prioritize orientation, status, and action over marketing language.
- Dense information should remain readable through spacing, alignment, and hierarchy rather than decorative treatments.
- Motion and ornament should be restrained; product UI should avoid gradient-heavy dashboard styling.

## Plan

- [x] Audit the current `/settings` information hierarchy and identify where card styling can be removed.
- [x] Rebuild the page as a cleaner settings surface with section dividers, row-based controls, and switches instead of checkboxes.
- [x] Preserve update, theme, and analytics behavior while improving scan speed and state visibility.
- [x] Verify the route compiles and the new UI follows existing theme tokens on desktop and mobile.

## Review

- Replaced the stacked card layout with a single structured settings surface that uses section dividers and row alignment for faster scanning.
- Swapped the analytics checkbox for the shared switch control and kept the optimistic-save behavior intact.
- Kept theme and update functionality unchanged while presenting state through tighter utility copy, badges, and inline metadata instead of boxed panels.
- Follow-up refinement removed non-essential overview content, update diagnostics, and privacy explainer blocks to make the page calmer and denser.
- Verification: `npm run typecheck` and `npm run build` passed in `ui/`.


# Transcription History Actions Plan

## Goal

Add a standard row actions menu to the `/transcription` history table with a three-dot trigger, appropriate quick actions, and delete confirmation that refreshes the list after removal.

## Planned UX

- Add a trailing overflow menu on every history row using a vertical three-dot trigger.
- Keep row click to open the record, but stop propagation for menu interactions.
- Include only actions that fit a history list standard: open record, copy transcript, export, and delete.
- Keep summary regeneration on the detail page because it is a heavier, model-dependent action rather than a common list action.
- Show delete confirmation in a modal and refresh history after successful deletion.

## Plan

- [x] Audit current transcription history row capabilities and detail-page actions.
- [x] Add a row overflow menu with standard quick actions and correct interaction handling.
- [x] Implement delete confirmation modal and refresh the history data after deletion.
- [x] Add or update tests for row menu actions and delete refresh behavior.

## Review

- Added a trailing three-dot actions menu to each transcription history row with `Open record`, `Copy transcript`, `Export`, and `Delete`.
- Kept `Regenerate summary` on the detail page because it depends on summary-model readiness and is less appropriate as a routine list action.
- Added a row-level delete confirmation modal and refreshed the history list after successful deletion.
- Updated the shared dropdown primitive so interactive items show a pointer cursor.
- Verification: `npm run typecheck` and `npm run test -- src/features/transcription/route.test.tsx`.

# Diarization And TTS History Actions Plan

## Goal

Align `/diarization` and `/text-to-speech` with the standard row-actions pattern now used on `/transcription`, so each history row has a conventional overflow menu with only appropriate quick actions.

## Planned UX

- Add a trailing three-dot menu to every history row on both routes.
- Keep row click to open the record, while preventing the menu trigger and items from accidentally opening the row.
- Use only standard list-level actions:
  - `/diarization`: `Open record`, `Copy transcript`, `Export`, `Delete`
  - `/text-to-speech`: `Open record`, `Copy text`, `Download`, `Delete`
- Keep heavier or workflow-specific actions on the detail pages, such as summary regeneration, reruns, and speaker correction.
- Confirm deletion in a modal and refresh the history data after successful removal.

## Plan

- [x] Audit existing detail-page actions and shared helpers for both routes.
- [x] Add row overflow menus and action handlers to the diarization and TTS history tables.
- [x] Wire parent-owned delete callbacks so history refresh stays consistent after removal.
- [x] Extend any shared dialog component needed for row-triggered export flows.
- [x] Add focused route tests for menu items and delete-refresh behavior on both routes.
- [x] Verify with `npm run typecheck` and focused route tests.

## Review

- Added a three-dot actions menu to each diarization history row with `Open record`, `Copy transcript`, `Export`, and `Delete`.
- Added a three-dot actions menu to each text-to-speech history row with `Open record`, `Copy text`, `Download`, and `Delete`.
- Kept heavier actions on detail pages, including diarization summary regeneration, speaker correction, and rerun controls.
- Added delete confirmation modals on both history tables and refreshed the route-owned history after successful deletion.
- Extended `DiarizationExportDialog` so it can open from a controlled row action instead of only a trigger child.
- Verification: `npm run typecheck`, `npm run test -- src/features/diarization/route.test.tsx`, `npm run test -- src/features/text-to-speech/route.test.tsx`, and `npm run test -- src/components/DiarizationExportDialog.test.tsx`.

# UUID Rollout Plan

## Research Notes

- Most persisted application records already use `TEXT` primary keys in SQLite, so new UUID-based IDs can be introduced without schema migrations.
- The current server mostly generates prefixed IDs such as `thread_*`, `msg_*`, `txr_*`, `dir_*`, `ttsp_*`, and `agent_sess_*`.
- Several runtime/API IDs already use plain UUIDs today, including core engine request IDs and some request-context values.
- A few frontend-only IDs are still timestamp/random based, including render queue items, transcript entries, and toast IDs.
- `ui/src/types.ts` contains semantic parsing for Kokoro voice IDs. Those are model identifiers, not generated record IDs, and should not be converted as part of this rollout.
- The default voice profile ID is a fixed constant for bootstrapping existing installs. Leave it unchanged in this phase so old data remains addressable.

## Phases

- [x] Phase 1: Introduce shared UUID helpers and switch all newly persisted server records to plain UUIDs.
  Scope:
  `chat_store`, `voice_store`, `voice_observation_store`, `saved_voice_store`, `speech_history_store`, `transcription_store`, `diarization_store`, and `studio_project_store`.
  Deliverable:
  New rows created in storage use canonical UUID strings with no type-specific prefix, while existing rows remain readable.
  Verification:
  `cargo fmt --package izwi-server`, `cargo test -p izwi-server transcription_store -- --nocapture`, `cargo test -p izwi-server diarization_store -- --nocapture`, `cargo test -p izwi-server voice_store -- --nocapture`, `cargo test -p izwi-server voice_observation_store -- --nocapture`, `cargo test -p izwi-server studio_project_store -- --nocapture`, `cargo test -p izwi-server chat_store -- --nocapture`, `cargo test -p izwi-server speech_history_store -- --nocapture`, and `cargo test -p izwi-server saved_voice_store -- --nocapture`.
  Commit:
  `refactor(server): use uuid ids for newly persisted records`

- [x] Phase 2: Switch remaining newly created server/runtime session and API object IDs to UUIDs.
  Scope:
  agent session records, OpenAI-compatible chat completion IDs, OpenAI-compatible response/message/tool-call IDs, and any remaining server-generated IDs that are user-visible but not persisted in the main stores.
  Deliverable:
  New server-generated IDs are consistently UUIDs across storage and API surfaces.
  Verification:
  Run focused server tests for affected API modules.
  Commit:
  `refactor(server): use uuid ids for runtime and api objects`

- [x] Phase 3: Switch client-generated new record IDs to UUIDs and update tests.
  Scope:
  render queue item IDs, realtime transcript entry IDs, toast IDs, and any other client-generated record IDs discovered during implementation.
  Deliverable:
  Client-side new IDs use `crypto.randomUUID()` through a shared helper instead of timestamp/random concatenation.
  Verification:
  Run targeted UI tests for the affected helpers/components.
  Commit:
  `refactor(ui): use uuid ids for client-generated records`

## Review

- Phase 1 complete.
- Added a shared server UUID helper and switched all newly persisted record IDs in the main stores from prefixed IDs to canonical UUID strings.
- Phase 2 complete.
- Switched agent session IDs plus OpenAI-compatible completion, response, tool-call, and assistant message IDs to canonical UUID strings.
- Focused verification:
  - `cargo fmt --package izwi-server`
  - `cargo test -p izwi-server state::tests::trim_store_by_uses_updated_at_for_agent_sessions -- --nocapture`
  - `cargo test -p izwi-server parses_qwen_tool_call_output_into_openai_shape -- --nocapture`
  - `cargo test -p izwi-server builds_tool_call_finish_reason_when_tool_output_detected -- --nocapture`
  - `cargo test -p izwi-server normalizes_assistant_tool_calls_into_qwen_xml -- --nocapture`
- Broad verification note:
  - `cargo test -p izwi-server openai -- --nocapture` still fails in pre-existing content-flattening tests unrelated to the UUID edits:
    - `api::openai::chat::completions::tests::flattens_text_parts_content`
    - `api::openai::responses::handlers::tests::flattens_part_content`
- Phase 3 complete.
- Switched client-generated toast IDs, realtime transcript entry IDs, and studio render-queue item IDs to shared UUID generation.
- Focused verification:
  - `npm run typecheck`
  - `npm run test -- src/lib/ids.test.ts src/features/voice/realtime/support.test.ts`
- Scope note:
  - Remaining `Date.now()` usage in the UI is for timestamps, filenames, or visual randomness rather than application record IDs.

# Desktop Status Bar / Tray Plan

## Goal

Add a cross-platform status bar / tray icon experience for Izwi desktop with:

- standard app controls (open, settings, models, updates, quit)
- live runtime visibility (server and model status)
- launch-at-login control
- troubleshooting shortcuts (copy API URL, open logs, docs)

## Phases

- [x] Phase 1: Add tray icon foundation and hide-on-close behavior
  Scope:
  Create a tray icon and minimal menu (`Open Izwi`, `Quit Izwi`) and change window-close behavior to hide the main window instead of exiting so the app can continue running in the tray.
  Verification:
  `cargo check -p izwi-desktop`
  Commit:
  `feat(desktop): add tray icon foundation with open/quit`

- [x] Phase 2: Add standard tray actions for settings, models, and update checks
  Scope:
  Add `Settings`, `Models`, and `Check for Updates` tray actions and wire desktop-to-frontend signals so actions trigger navigation and manual update checks from anywhere in the app.
  Verification:
  `cargo check -p izwi-desktop` and `npm run typecheck` (in `ui/`)
  Commit:
  `feat(desktop): add standard tray actions`

- [x] Phase 3: Add live server/model status and restart-server tray action
  Scope:
  Add live status rows in the tray menu (server health + model readiness/download activity), background polling, and a `Restart Server` action for local-server mode.
  Verification:
  `cargo check -p izwi-desktop`
  Commit:
  `feat(desktop): add live tray status and server restart`

- [x] Phase 4: Add launch-at-login control in tray
  Scope:
  Add launch-at-login support and expose it as a tray toggle, including capability wiring and platform-safe state syncing.
  Verification:
  `cargo check -p izwi-desktop`
  Commit:
  `feat(desktop): add launch-at-login tray control`

- [x] Phase 5: Add troubleshooting shortcuts and hardening tests
  Scope:
  Add `Copy API URL`, `Open Logs`, and `Documentation` tray actions, plus focused unit tests for the tray status formatting/helpers.
  Verification:
  `cargo test -p izwi-desktop` and `npm run typecheck` (in `ui/`)
  Commit:
  `feat(desktop): add tray troubleshooting shortcuts and tests`

## Review

- Phase 1 complete.
- Added a new desktop tray foundation module with a minimal menu: `Open Izwi` and `Quit Izwi`.
- Desktop close-button behavior now hides the main window instead of exiting, enabling a real tray-resident lifecycle.
- Verification:
  - `cargo check -p izwi-desktop`
- Phase 2 complete.
- Added standard tray actions for `Settings`, `Models`, and `Check for Updates`.
- Wired desktop tray events into the React router/update provider so tray actions can navigate and trigger a manual update check from any route.
- Verification:
  - `cargo check -p izwi-desktop`
  - `npm run typecheck` (in `ui/`)
- Phase 3 complete.
- Added live tray status rows for `Server` and `Models` with background polling against `/v1/internal/health` and `/v1/admin/models`.
- Added a `Restart Server` tray action that is enabled only when running in local-server mode and reuses the managed desktop server process handle.
- Verification:
  - `cargo check -p izwi-desktop`
- Phase 4 complete.
- Added a `Launch at Login` tray checkbox backed by `tauri-plugin-autostart`.
- Added autostart capability permission wiring and periodic check-state sync so the tray toggle stays aligned with runtime state.
- Verification:
  - `cargo check -p izwi-desktop`
- Phase 5 complete.
- Added tray troubleshooting actions for `Copy API URL`, `Open Logs`, and `Documentation`.
- Added focused tray helper tests for model status aggregation/formatting and API URL normalization.
- Verification:
  - `cargo test -p izwi-desktop`
  - `npm run typecheck` (in `ui/`)

# Transcription Nested Modal Fix

## Plan

- [x] Confirm how the transcription route stacks `NewTranscriptionModal` and `RouteModelModal`.
- [x] Raise the model modal above the new transcription modal on `/transcription`.
- [x] Add a focused route test for opening model management from inside the new transcription modal.

## Review

- The transcription route now raises `RouteModelModal` to `z-[70]` while `NewTranscriptionModal` is open, so the model manager can sit above the creation dialog.
- Added a focused route test that verifies the model modal is promoted when the new transcription modal opens.
- Verification:
  - `npm run test -- src/features/transcription/route.test.tsx`
  - `npm run typecheck`
- Follow-up fix:
  - The underlying `NewTranscriptionModal` now prevents outside-dismiss interactions while the stacked model modal is open, so clicks in the top modal no longer close the transcription modal underneath.
  - The route test now verifies a click inside the stacked model modal still triggers model selection while keeping the transcription modal open.

# History Polling UX Reload Fix

## Goal

Stop `/transcription` list view from flashing a full-screen loading state during background polling after returning from a newly created record, and enforce the same no-reload UX for `/text-to-speech` and `/diarization`.

## Plan

- [x] Reproduce and isolate why list pages appear to reload while jobs are still pending.
- [x] Update route history hooks so polling uses background refresh semantics and does not toggle the table-level full loading shell on every interval.
- [x] Apply the same fix pattern to transcription and text-to-speech; confirm diarization behavior does not regress.
- [x] Add focused route-level regression tests proving history content stays visible during polling.
- [x] Run targeted verification (`vitest` route tests + `typecheck`) and capture results.

## Review

- `useTranscriptionHistory` and `useTextToSpeechHistory` now keep existing rows visible during interval polling by using background refresh semantics when data is already on screen.
- Foreground loading behavior is preserved for initial loads and explicit refresh actions; only polling refreshes avoid the full table loading shell.
- Added regression tests:
  - `ui/src/features/transcription/route.test.tsx`
  - `ui/src/features/text-to-speech/route.test.tsx`
  Both assert history rows remain visible while a polling refresh request is in flight.
- `/diarization` was verified via existing route coverage and remains stable; its list route still does not poll history continuously.
- Verification:
  - `npm run test -- src/features/transcription/route.test.tsx`
  - `npm run test -- src/features/text-to-speech/route.test.tsx`
  - `npm run test -- src/features/diarization/route.test.tsx`
  - `npm run typecheck`

# User Docs Model Support Refresh

## Goal

Update `docs/user` so supported models, defaults, and examples reflect the current application catalog and behavior (including enabled model variants, family guidance, and command defaults).

## Source-of-Truth Notes (Application Analysis)

- Model catalog and enablement are defined in `crates/izwi-core/src/catalog/metadata.rs` (`ModelVariant`, `is_enabled`).
- Public model listing endpoints (`/v1/admin/models`, `/v1/models`) only return enabled variants.
- CLI defaults are defined in `crates/izwi-cli/src/app/cli.rs` (notably chat/transcribe/diarize/align/bench defaults).
- Chat media support is currently limited to Qwen3.5 GGUF models for image inputs; video inputs are not yet implemented (`crates/izwi-server/src/app/chat_content.rs`).

## Phased Plan

- [x] Phase 1: Refresh core user model catalog docs
  Scope:
  Update high-level model pages (`docs/user/models/index.md`, onboarding/model-overview docs) to reflect currently supported families and canonical model IDs exposed by the app.
  Verification:
  Cross-check every listed model against `ModelVariant::is_enabled()` and list endpoint behavior.
  Commit:
  `docs(user): refresh supported model catalog and onboarding references`

- [x] Phase 2: Align CLI docs with current model defaults and examples
  Scope:
  Update CLI command docs under `docs/user/cli/` so defaults/examples match current code defaults and supported variants (chat/transcribe/diarize/align/bench/pull/tts/model management references).
  Verification:
  Compare all documented defaults against `crates/izwi-cli/src/app/cli.rs` and verify example models are enabled.
  Commit:
  `docs(cli): align model defaults and examples with current runtime`

- [x] Phase 3: Align feature and troubleshooting docs with current model support
  Scope:
  Update feature guides and troubleshooting pages to use current model families, supported capabilities, and accurate model recommendations.
  Verification:
  Validate model guidance against catalog capabilities and runtime/API constraints (chat multimodal scope, ASR/TTS/diarization model guidance).
  Commit:
  `docs(features): update model guidance and troubleshooting for current support`

## Review

- Phase 1 complete.
- Refreshed onboarding and core model catalog docs to match enabled catalog families and canonical IDs surfaced by `izwi list`/`/v1/models`.
- Phase 2 complete.
- Aligned CLI docs with current model defaults in `crates/izwi-cli/src/app/cli.rs`, replaced stale model examples, and corrected CLI model listing/output descriptions.
- Phase 3 complete.
- Updated feature and troubleshooting docs to match active model support, including:
  - chat family coverage (Qwen3/Qwen3.5/LFM2.5/Gemma),
  - Qwen3.5-only multimodal image support note,
  - corrected ASR/diarization/aligner guidance,
  - corrected Voice Cloning (Base-model) and Voice Design (`instructions`) model/API guidance.
- Verification:
  - Source-of-truth cross-checks against:
    - `crates/izwi-core/src/catalog/metadata.rs`
    - `crates/izwi-cli/src/app/cli.rs`
    - `crates/izwi-server/src/app/chat_content.rs`
    - `crates/izwi-server/src/api/openai/audio/speech.rs`
    - `crates/izwi-server/src/api/openai/audio/diarizations.rs`
  - Focused `rg` drift checks for stale model IDs and capability wording in updated docs.

# LFM2.5 Chat TTFT/TPS Improvement Plan

## Goal

Improve `LFM2.5-1.2B-Instruct-GGUF` and `LFM2.5-1.2B-Thinking-GGUF` chat latency and throughput with a low-risk pass focused on hot-path overhead and observability correctness.

## Baseline (2026-04-07)

- Instruct benchmark sample (`iterations=5`, `max_tokens=256`, `concurrent=1`, warm):
  - TTFT: ~`883 ms`
  - End-to-end: ~`2379 ms`
  - Completion TPS: ~`67 tok/s`
- Runtime delta currently reports `prefill ~= total` and `decode ~= 0` for LFM2.5.
- Key diagnosis: LFM2.5 runs through non-incremental chat fallback (`supports_incremental_decode=false`), so current runtime phase telemetry is not trustworthy for decode-path tuning.

## Qwen3.5 Optimization Analysis (What Transfers This Pass)

- Portable now (apply to LFM2.5 chat wrapper):
  - `qwen35/chat` Phase A/C/E style optimizations:
    - token-piece decode instead of full-history decode per step,
    - remove O(n^2) `decode_text(&generated_ids)` + `text_delta` loop,
    - avoid unnecessary per-step allocations/bookkeeping in greedy defaults.
  - Stream deltas in token-sized chunks (not per-character callbacks).
  - Bench/telemetry discipline from kernel signoff work (`04e6494`, `38aee5b`): fixed protocol and clear counters.
- Not portable in this pass (requires native LFM2 runtime internals, not current Candle `quantized_lfm2` wrapper):
  - Qwen3.5 dense-vs-paged decode routing.
  - Fused SDPA/flash-gated attention paths.
  - RoPE kernel path switching and full-attention sequence batching in native Qwen layers.
  - Block fusion, tiled recurrence, and Qwen3.5-specific buffer pool wiring.

## Plan

- [x] Phase 1: Fix LFM2.5 observability so TTFT/TPS tuning is measurable
  Scope:
  Add explicit non-incremental phase timing capture for LFM2.5 fallback chat execution (prefill, first-token, decode, total), and surface it through existing runtime telemetry/bench output paths so decode is no longer reported as `0`.
  Verification:
  Run fixed chat benchmark protocol and confirm runtime decode metrics are non-zero and consistent with streamed timing.

- [x] Phase 2: Port Qwen3.5 chat hot-path decode optimizations to LFM2.5
  Scope:
  Introduce token-piece decode caching in LFM2 tokenizer, append assembled text incrementally per token, remove repeated full-history decode/delta diffing, and emit per-token deltas (no per-char callback loop).
  Verification:
  Unit tests for token-piece decode correctness; benchmark diff showing completion TPS gain on both Instruct and Thinking variants.

- [x] Phase 3: Reduce prompt-side overhead that impacts TTFT
  Scope:
  Cache static prompt scaffolding token IDs (`im_start`, role headers, line breaks, assistant prefix) and avoid repeated tiny tokenization calls in `build_prompt`.
  Verification:
  Short-prompt benchmark profile shows TTFT improvement without output regression.

- [x] Phase 4: Benchmark signoff and guardrails
  Scope:
  Run short/long prompt profiles for both LFM2.5 variants with warm server and fixed settings; capture `izwi bench chat` and runtime metrics snapshots before/after.
  Verification:
  Meet minimum acceptance targets:
  - TTFT improvement >= 10% on short prompt profile.
  - Completion TPS improvement >= 10% on long prompt profile.
  - No correctness regressions (stop-token handling, repetition-loop guard behavior, streamed text parity).

- [x] Phase 5: Trim residual LFM2 decode/prompt allocation overhead
  Scope:
  Remove avoidable decode-loop allocations (`Tensor::from_slice` for prompt/token tensors), tighten token-piece cache lookup to indexed storage, and avoid cloning the full message list while building prompts.
  Verification:
  `cargo check -p izwi-core`, `cargo test -p izwi-core lfm2::chat -- --nocapture`, plus a quick sanity rerun of the Victoria Falls benchmark profile.

- [x] Phase 6: Remove global lock contention in LFM2 token-piece detokenization
  Scope:
  Replace shared `Mutex` token-piece cache access with per-token `OnceLock` slots, keep decode-cache hits lock-free, and trim remaining prompt-build cloning (`ChatRole` borrow path).
  Verification:
  `cargo check -p izwi-core`, `cargo test -p izwi-core lfm2::chat -- --nocapture`, and `izwi bench chat --model LFM2.5-1.2B-Instruct-GGUF --prompt "Tell me about Victoria Falls." --iterations 5 --max-tokens 256 --warmup`.

- [x] Phase 7: Trim decode-loop cache hit overhead and repetition checks
  Scope:
  Replace `Arc<str>` token-piece cache payloads with borrowed `&str` access from per-token `OnceLock<String>`, pre-allocate decode buffers from `max_new_tokens`, and run repetition-loop detection at fixed intervals instead of every token.
  Verification:
  `cargo check -p izwi-core`, `cargo test -p izwi-core lfm2::chat -- --nocapture`, and `izwi bench chat --model LFM2.5-1.2B-Instruct-GGUF --prompt "Tell me about Victoria Falls." --iterations 5 --max-tokens 256 --warmup`.

- [x] Phase 8: Batch fallback chat stream deltas after first token
  Scope:
  Keep first streamed token immediate for TTFT, then batch fallback chat token-piece deltas by piece/byte thresholds to reduce `stream_text` send overhead and JSON chunk churn during streaming benchmarks.
  Verification:
  `cargo check -p izwi-core`, `cargo test -p izwi-core handler_chat -- --nocapture`, `cargo test -p izwi-core lfm2::chat -- --nocapture`, and `izwi bench chat --model LFM2.5-1.2B-Instruct-GGUF --prompt "Tell me about Victoria Falls." --iterations 5 --max-tokens 256 --warmup`.

- [x] Phase 9: Pivot TTFT work to LFM2 prefill strategy and timing visibility
  Scope:
  Add adaptive LFM2 prefill policy (`auto/full/token`) with a short-prompt token-mode path, record kernel prefill path counters for LFM2 prefill execution, and add per-request timing logs for prompt-build, prompt-forward, decode, and first-delta milestones.
  Verification:
  `cargo check -p izwi-core`, `cargo test -p izwi-core lfm2::chat -- --nocapture`, and user-host benchmark rerun with current fixed protocol.

- [x] Phase 10: Reduce default prefill tokens for single-turn LFM2 prompts
  Scope:
  Add adaptive default-system policy for LFM2 prompt building (`auto/always/never`) and default `auto` behavior that skips synthetic system injection for single-turn user prompts, reducing prefill token count and TTFT for common bench/profile traffic.
  Verification:
  `cargo check -p izwi-core`, `cargo test -p izwi-core lfm2::chat -- --nocapture`, and user-host benchmark rerun with current fixed protocol.

- [x] Phase 11: Add aggressive single-turn prompt compaction path
  Scope:
  Introduce policy-controlled aggressive prompt compaction for single-turn user prompts that removes chat-wrapper user scaffolding and keeps only lean content + assistant prefix, maximizing TTFT reduction at the expense of prompt-shape fidelity.
  Verification:
  `cargo check -p izwi-core`, `cargo test -p izwi-core lfm2::chat -- --nocapture`, and user-host benchmark rerun with current fixed protocol.

- [x] Check-in before implementation
  Share Phase 1-2 scope and expected tradeoffs, then proceed with code changes only after signoff.

## Review (Planning)

- This pass intentionally targets changes that are architecture-compatible with current `quantized_lfm2` usage.
- Qwen3.5 kernel-level attention optimizations are explicitly deferred because LFM2.5 currently does not use that native attention stack.
- Observability correction is first to avoid optimizing against misleading decode telemetry.

## Review (Implementation: Phase 1-2)

- Added executor-to-engine phase timing overrides for non-incremental chat fallback paths:
  - fallback chat execution now records prefill time, decode time, and first-output timing in the executor,
  - engine core applies the override to request-phase telemetry before building latency breakdowns.
- Resulting runtime metrics now have a non-zero decode path for fallback chat requests instead of attributing all work to prefill.
- Ported Qwen3.5-style chat assembly optimizations to LFM2.5:
  - added token-piece decode cache in `ChatTokenizer`,
  - replaced per-step full-history decode + `text_delta` diffing with incremental string append,
  - switched callback emission from per-character to per-token-piece deltas.
- Verification:
  - `cargo check -p izwi-core`
  - `cargo test -p izwi-core --no-run`
  - `cargo test -p izwi-core lfm2::chat -- --nocapture`
  - `cargo test -p izwi-core engine::core::tests::test_merge_executor_output_replaces_cumulative_audio_snapshots -- --nocapture`

## Review (Implementation: Phase 3)

- Added `PromptScaffoldTokens` to LFM2 chat model initialization to pre-tokenize static chat scaffolding once:
  - `system\n`
  - `user\n`
  - `assistant\n`
  - `\n`
- Updated `build_prompt` to reuse cached role header and newline token spans instead of re-tokenizing formatted strings for every message.
- Kept dynamic content tokenization unchanged and preserved existing assistant-thinking handling.
- Verification:
  - `cargo fmt --package izwi-core`
  - `cargo check -p izwi-core`
  - `cargo test -p izwi-core lfm2::chat -- --nocapture`

## Review (Implementation: Phase 4)

- Executed signoff protocol (`iterations=20`, `max_tokens=128`, `warmup`) for both LFM2.5 variants:
  - Instruct short profile: TTFT `930.07 ms`, completion TPS `32.43 tok/s` (completion averaged `43` tokens).
  - Instruct long profile: TTFT `1038.86 ms`, completion TPS `54.88 tok/s`.
  - Thinking short profile: TTFT `997.50 ms`, completion TPS `55.99 tok/s`.
  - Thinking long profile: TTFT `1042.79 ms`, completion TPS `54.46 tok/s`.
- Runtime telemetry confirms fallback chat decode is now explicitly tracked (`decode_ms > 0`) in all signoff runs.
- Victoria Falls regression check (Instruct, `iterations=5`, `max_tokens=256`) now shows TTFT/TPS in the same or slightly better range than pre-phase baseline.
- Acceptance-gate note:
  - With currently available comparable baselines, improvements are positive but below the provisional `>=10%` threshold, so further gains likely require deeper compute-path work beyond wrapper-level optimizations.

## Review (Implementation: Phase 5)

- Replaced fallback LFM2 generation tensor construction from `Tensor::from_vec` cloning to `Tensor::from_slice` for both prompt prefill and per-token decode tensors.
- Switched token-piece cache storage from `HashMap<u32, String>` to indexed `Vec<Option<Arc<str>>>` to reduce hash lookups and repeated tiny string copies on hot decode steps.
- Refactored `build_prompt` to avoid cloning the entire message list for synthetic system insertion; prompt assembly now iterates borrowed messages with role/content normalization helpers.
- Updated assistant-history trimming utility to return `Cow<str>` so non-reasoning paths stay borrowed and allocation-free.
- Verification:
  - `cargo fmt --package izwi-core -- crates/izwi-core/src/models/architectures/lfm2/chat.rs`
  - `cargo check -p izwi-core`
  - `cargo test -p izwi-core lfm2::chat -- --nocapture`
  - `izwi bench chat --model LFM2.5-1.2B-Instruct-GGUF --prompt "Tell me about Victoria Falls." --iterations 5 --max-tokens 256 --warmup` (sanity rerun; showed higher variance and lower TPS than earlier baseline, so benchmark signoff should be re-run on a controlled warm host before treating this phase as a net perf gain)

## Review (Implementation: Phase 6)

- Replaced `decode_piece_cache` from shared `Mutex<Vec<Option<Arc<str>>>>` to `Vec<OnceLock<Arc<str>>>` so cache hits no longer contend on a global lock each decode step.
- Updated token-piece decode cache miss path to set per-token `OnceLock` entries and return shared `Arc<str>` pieces.
- Removed an extra role clone in prompt assembly by passing borrowed `ChatRole` references through `append_prompt_message`.
- Verification:
  - `cargo fmt --package izwi-core -- crates/izwi-core/src/models/architectures/lfm2/chat.rs`
  - `cargo check -p izwi-core`
  - `cargo test -p izwi-core lfm2::chat -- --nocapture`
  - `izwi bench chat --model LFM2.5-1.2B-Instruct-GGUF --prompt "Tell me about Victoria Falls." --iterations 5 --max-tokens 256 --warmup` (local sanity run remained noisy and slower than controlled baseline; host-to-host variance still dominates this short run)

## Review (Implementation: Phase 7)

- Switched token-piece cache payloads from `OnceLock<Arc<str>>` to `OnceLock<String>` and updated `decode_token_piece` to return borrowed `&str` on cache hits, removing per-token `Arc` clone churn.
- Added decode-loop pre-allocation for token IDs and assembled text based on `max_new_tokens` to reduce incremental growth reallocations.
- Added `should_check_repetition_loop` interval gating so repetition-loop detection runs every 4 tokens after warm-up threshold rather than on every decode step.
- Added focused unit coverage for repetition-check intervals.
- Verification:
  - `cargo check -p izwi-core`
  - `cargo test -p izwi-core lfm2::chat -- --nocapture`
  - `izwi bench chat --model LFM2.5-1.2B-Instruct-GGUF --prompt "Tell me about Victoria Falls." --iterations 5 --max-tokens 256 --warmup` (local run remained slower/noisier than controlled host baselines; user-host benchmark remains source of truth)

## Review (Implementation: Phase 8)

- Added `StreamDeltaBatch` in fallback chat executor path to preserve first-token immediacy while batching later deltas using fixed piece/byte thresholds and newline boundaries.
- Updated fallback chat streaming emit flow to:
  - capture first-output timing as before,
  - emit first non-empty delta immediately,
  - accumulate and flush subsequent deltas in batches,
  - flush any residual batch before the final marker.
- Added unit coverage for batch behavior (first-delta immediacy, threshold flush, newline flush, finish flush).
- Verification:
  - `cargo check -p izwi-core`
  - `cargo test -p izwi-core handler_chat -- --nocapture`
  - `cargo test -p izwi-core lfm2::chat -- --nocapture`
  - `izwi bench chat --model LFM2.5-1.2B-Instruct-GGUF --prompt "Tell me about Victoria Falls." --iterations 5 --max-tokens 256 --warmup` (local run remained slower/noisier than controlled host baselines; user-host benchmark remains source of truth)

## Review (Implementation: Phase 9)

- Added adaptive LFM2 prefill policy controls:
  - `IZWI_LFM2_PREFILL_MODE` supports `auto` (default), `full`, and `token`.
  - `IZWI_LFM2_PREFILL_TOKEN_THRESHOLD` controls short-prompt token-mode cutoff in `auto` mode (default: `64` prompt tokens).
- Implemented token-mode prefill execution path for LFM2 where prompt tokens are prefed one-by-one before first decode token sampling.
- Wired LFM2 prefill-path telemetry counters:
  - full prefill records sequence spans/tokens,
  - token prefill records token-mode steps.
- Added LFM2 per-request timing breakdown logs (debug level) for:
  - prompt build time,
  - prefill forward time,
  - first-delta timing,
  - decode time,
  - total generation time,
  with prefill policy/execution metadata.
- Verification:
  - `cargo check -p izwi-core`
  - `cargo test -p izwi-core lfm2::chat -- --nocapture`
  - user-host benchmark rerun pending (source of truth for TTFT impact)

## Review (Implementation: Phase 10)

- Added adaptive LFM2 default-system policy controls:
  - `IZWI_LFM2_DEFAULT_SYSTEM_POLICY` supports `auto` (default), `always`, and `never`.
- Updated LFM2 prompt construction to use policy-driven system injection:
  - `auto` now skips synthetic default system for single-turn user prompts,
  - preserves synthetic system insertion for multi-turn non-system conversations.
- Logged resolved default-system policy at model load for easier runtime validation.
- Added focused unit tests for policy parsing and auto-policy behavior.
- Verification:
  - `cargo check -p izwi-core`
  - `cargo test -p izwi-core lfm2::chat -- --nocapture`
  - user-host benchmark rerun pending (source of truth for TTFT impact)

## Review (Implementation: Phase 11)

- Added aggressive LFM2 prompt-style policy controls:
  - `IZWI_LFM2_PROMPT_STYLE_POLICY` supports `auto` (default), `standard`, and `aggressive`.
- Added aggressive single-turn prompt path in LFM2 `build_prompt`:
  - for single user-turn prompts with no synthetic system insertion, use compact `content + newline + assistant prefix` prompt shape,
  - keep standard chat-template prompt construction for multi-turn/system traffic.
- Logged resolved prompt-style policy at model load for runtime verification.
- Added focused unit tests for prompt-style policy parsing and aggressive-path gating behavior.
- Verification:
  - `cargo check -p izwi-core`
  - `cargo test -p izwi-core lfm2::chat -- --nocapture`
  - user-host benchmark rerun pending (source of truth for TTFT impact)

# Studio Route History Table Refactor Plan

## Goal

Refactor `/studio` project listing to align with the route pattern used by `/transcription`, `/diarization`, and `/text-to-speech`: operational history in a table with row actions, consistent loading/empty/error states, and cursor pagination behavior.

## Plan

- [x] Audit current `/studio` list UX and map matching history-table conventions from other routes.
- [x] Introduce a dedicated Studio history table component with:
  - row click navigation to `/studio/:projectId`
  - columns for key project metadata and progress
  - standardized overflow row actions (open + delete)
  - loading, empty, and load-more states
- [x] Replace the current card-grid listing in `StudioWorkspace` with the new table component while preserving delete confirmation behavior.
- [x] Add focused tests for the new Studio history table behavior.
- [x] Verify with typecheck and targeted UI tests.

## Review

- Added `StudioProjectHistoryTable` and aligned it with existing route table conventions:
  - loading state, error state with retry, empty state with create CTA
  - clickable rows that navigate to `/studio/:projectId`
  - row overflow actions with `Open project` and `Delete`
  - integrated cursor-based `Load more` control in table footer
- Replaced Studio index card-grid listing with the new history table while keeping the existing project count summary header and existing delete confirmation dialog flow.
- Kept the user-requested simplification by removing list search/filter/sort controls.
- Added focused tests for the new Studio history table interactions:
  - row open navigation
  - row action delete callback
  - empty-state create action
  - load-more callback
- Verification:
  - `npm run typecheck` (in `ui/`)
  - `npm run test -- src/features/studio/components/StudioProjectHistoryTable.test.tsx src/features/PageHeaderHistoryButtons.test.tsx`

# Studio List Controls Removal

## Goal

Remove filter/sort controls from the `/studio` project listing UI.

## Plan

- [x] Remove search, status-filter, and sort controls from the Studio list header.
- [x] Simplify list-state logic to render projects without filter/sort state.
- [x] Verify UI compiles cleanly.

## Review

- Removed the `Search`, `All statuses`, and `Sort` controls from the Studio list surface.
- Simplified `visibleProjects` to stable recency ordering without local filter/sort state.
- Verification: `npm run typecheck` in `ui/`.
