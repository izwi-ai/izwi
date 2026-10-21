# Releasing Izwi

Releases are tag-driven.
Pushing a tag that matches `v*` triggers `.github/workflows/release.yml` automatically.
There is no manual trigger in the workflow right now.

## Files to update for a new version

The version lives in two files; the release workflow checks that the tag
matches both before it builds anything:

1. `Cargo.toml`: `[workspace.package].version`. The desktop app
   (`crates/izwi-desktop/tauri.conf.json` has no version), the CLI and the
   downloader's User-Agent all follow it.
2. `ui/package.json` and the root of `ui/package-lock.json`: the same version.

Then run `cargo check --workspace` so `Cargo.lock` picks it up, and refresh
the checked-in gateway OpenAPI document
(`crates/izwi-gateway/src/api/gateway-openapi.json`), which carries the
version too. The first run rewrites it and fails; the second confirms it:

```bash
IZWI_UPDATE_GATEWAY_OPENAPI=1 cargo test -p izwi-server --lib gateway_openapi_document_is_current
cargo test -p izwi-server --lib gateway_openapi_document_is_current
```

Version numbers are semver. Stable releases are `0.1.0`, `0.1.1`…;
prereleases add a label, `0.2.0-beta-1`. Beta numbers compare as numbers
(`beta-10` is newer than `beta-9`).

## Release steps (example: v0.1.1)

1. Update the version files listed above.
2. Run local checks:

```bash
cargo check --workspace
npm ci --prefix ui
npm --prefix ui run build
```

3. Commit the release prep changes:

```bash
git add Cargo.toml Cargo.lock ui/package.json ui/package-lock.json \
  crates/izwi-gateway/src/api/gateway-openapi.json
git commit -m "Prepare release v0.1.1"
```

4. Push the commit:

```bash
git push origin <your-branch>
```

5. Tag and push the release tag:

```bash
git tag -a v0.1.1 -m "Izwi v0.1.1"
git push origin v0.1.1
```

6. Verify release:
   - Open GitHub Actions and confirm the `Release` workflow runs for the tag.
   - Confirm the `Backend Truth` workflow is green on the branch or merge commit that produced the tag.
   - Open the GitHub Releases page and confirm all assets are attached,
     including the updater manifest (`latest.json`, or `latest-beta.json`
     for a prerelease) and the `.sig` files it points to.
   - Confirm the release body includes the runtime support note describing the CPU-only native artifacts and Docker/source CUDA paths.

## In-app updates

Every release is signed for the updater (`TAURI_SIGNING_PRIVATE_KEY`,
`TAURI_SIGNING_PRIVATE_KEY_PASSWORD`, and the matching `IZWI_UPDATER_PUBKEY`
that the workflow bakes into the app). The publish job writes the manifest:

- a stable tag (`v0.1.1`) publishes `latest.json` and is a normal release;
- a tag with a label (`v0.2.0-beta-1`) publishes `latest-beta.json` and is a
  prerelease.

A stable build follows stable releases only. A beta build follows betas and
stable releases, so testers move on to the release they tested. The app
reads GitHub's release list, takes the newest release its channel accepts
that carries the right manifest, and installs the entry for its platform:
`macos`, `windows`, `linux` (the AppImage) or `linux-deb` for an app
installed from the `.deb` (added when the build signed the `.deb`).

To try an update before publishing, point a build at a staged manifest:
`IZWI_UPDATER_FORCE_MANIFEST_URL=https://…/latest.json`.

### From the old betas to 0.1.0

Betas up to `v0.1.0-beta-17` only look at `v0.1.0-beta-N` tags with
`latest-beta.json`, so they never see a stable release. `v0.1.0-beta-18` is
their bridge: a normal beta release, built with the updater above. Users
go beta-17 → beta-18 → 0.1.0:

1. Release `v0.1.0-beta-18`. Old betas update to it; it stops the server an
   old beta left running.
2. Release `v0.1.0`. beta-18 follows stable releases too, so it offers 0.1.0.

Keep the beta-18 release published after 0.1.0 ships: it is the only way
from an old beta to 0.1.0.

## What the release workflow builds

For each OS runner, the workflow builds:

1. Terminal bundle:
   - `izwi` (CLI)
   - `izwi-server`
   - `izwi-desktop` (desktop shell binary used by `izwi serve --mode desktop`)
2. Desktop installer bundle:
   - Linux: `.deb`
   - Windows: `NSIS .exe`
   - macOS: `.dmg`

## Current backend truth for release artifacts

GitHub Releases currently publish:

1. macOS assets as the Metal-capable path on Apple Silicon with macOS 15+;
   those same assets fall back to CPU on macOS 12-14.
2. Linux and Windows assets as CPU-only native artifacts with CPU-safe public entrypoints.

Linux and Windows public binary names must remain unchanged:

1. `izwi` / `izwi-server` on Linux
2. `izwi.exe` / `izwi-server.exe` on Windows

Linux and Windows native release artifacts must not contain `runtime/cuda`, CUDA shared libraries, or CUDA DLLs. The supported NVIDIA paths are:

1. The Docker CUDA image/profile on NVIDIA Linux hosts
2. Source builds using `cargo build --release --features cuda`

The `Backend Truth` and `Release` workflows are the pre-release guardrails for this contract:

1. CPU cargo checks run separately from CUDA cargo checks.
2. CPU Docker builds run separately from CUDA Docker builds.
3. Linux and Windows release jobs verify native artifacts are CPU-only, reject CUDA runtime payloads, and enforce a conservative per-file size cap.
4. Docker build jobs smoke-run the final CPU image and audit the CUDA image's dynamic dependencies, allowing `libcuda.so.1` to come from the NVIDIA container runtime on GPU hosts.
5. A GPU-host runtime smoke test is still the next gap; until an NVIDIA runner exists, end-to-end CUDA execution remains a manual release check on an NVIDIA Linux host.

The desktop bundle build runs the UI build automatically via Tauri `beforeBuildCommand`, so `ui/dist` is rebuilt from source on each release run.

Linux `.deb` installs:

1. `/usr/bin/izwi`
2. `/usr/bin/izwi-server`

## Signing and notarization

A stable release must be signed and notarized on macOS: its build fails
without the Apple secrets. Prereleases build unsigned when they are missing.

The repository secrets:

1. `APPLE_CERTIFICATE`
2. `APPLE_CERTIFICATE_PASSWORD`
3. `APPLE_SIGNING_IDENTITY`
4. `APPLE_ID`
5. `APPLE_PASSWORD`
6. `APPLE_TEAM_ID`

## Troubleshooting workflow parse errors

If GitHub shows an error like:

- `Unrecognized named-value: 'secrets'` inside an `if:` expression

Cause:

- GitHub Actions does not allow direct `secrets.*` usage in `if:` conditions.

Fix:

- Map secrets into environment variables (`env`) and reference `env.*` in `if:`.

## macOS install/uninstall

For user-facing macOS instructions, see `docs/MACOS.md`.
